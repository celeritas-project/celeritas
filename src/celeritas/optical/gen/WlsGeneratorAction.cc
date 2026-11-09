//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/WlsGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "WlsGeneratorAction.hh"

#include "corecel/Assert.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/WavelengthShiftData.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/model/WavelengthShiftModel.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "WavelengthShiftGenerator.hh"

#include "detail/GeneratorAlgorithms.hh"
#include "detail/OffloadAlgorithms.hh"
#include "detail/WlsGeneratorExecutor.hh"

namespace celeritas
{
namespace optical
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Compact the buffer and count the distributions created during a step.
 *
 * Distributions created during the previous step were written past the
 * compacted buffer, at most one per track slot. This compacts them into the
 * buffer, computes the cumulative photon counts, and accumulates the new
 * distributions and photons. Since the buffer size is only known on device,
 * the buffer operations are launched over an upper bound of the buffer size,
 * which is returned.
 */
template<MemSpace M>
size_type update_buffer(WlsGeneratorState<M>& state)
{
    auto& buffer = state.store.ref().distributions;
    auto counters = state.device_counters();
    auto stream_id = state.stream_id();

    // Upper bound on the buffered distributions, including the ones created
    // in the previous step. The buffer was grown to hold these during the
    // previous step.
    size_type bound = state.buffer_bound() + state.num_slots;
    CELER_ASSERT(bound <= buffer.size());

    // Compact the buffer, storing the number of valid distributions, and
    // clear the unspecified data left past the compacted buffer
    detail::select_valid(buffer, bound, counters, stream_id);
    detail::clear_tail(buffer, bound, counters, stream_id);

    // Calculate the cumulative sum of the number of photons in the buffered
    // distributions, which determines which thread will generate photons
    // from which distribution. The new photons were already added to the
    // core state's pending count when the distributions were created.
    detail::inclusive_scan_photons(
        buffer, state.store.ref().offsets, bound, counters, stream_id);

    return bound;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Count and discard buffered distributions after an aborted loop.
 *
 * WLS interactions update the core pending count immediately, but their
 * distributions are normally accumulated at the next generation step. After
 * counting them, the entire buffer is cleared so that the leftover
 * distributions aren't generated in the next transport loop.
 */
template<MemSpace M>
void WlsGeneratorState<M>::reset_counters()
{
    update_buffer(*this);
    this->clear_buffer();
    GeneratorCounterState<M>::reset_counters();
}

template struct WlsGeneratorState<MemSpace::host>;
template struct WlsGeneratorState<MemSpace::device>;

//---------------------------------------------------------------------------//
/*!
 * Construct with IDs, WLS model, and initial buffer capacity.
 *
 * The second WLS model is optional. The distribution buffer grows as needed
 * during transport.
 */
WlsGeneratorAction::WlsGeneratorAction(Input&& input)
    : GeneratorBase(input.action_id,
                    input.aux_id,
                    input.gen_id,
                    "wls-generate",
                    "generate photons from a wavelength shifting process")
    , wls_(std::move(input.wls))
    , wls2_(std::move(input.wls2))
    , capacity_(input.capacity)
{
    CELER_EXPECT(capacity_ > 0);
    CELER_EXPECT(wls_ || wls2_);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto WlsGeneratorAction::create_state(
    MemSpace m, StreamId id, size_type size) const -> UPState
{
    // Each track slot can write a distribution past the end of the buffer
    // during a step, so the buffer must always hold at least one per slot
    size_type const capacity = std::max(capacity_, size);

    if (m == MemSpace::host)
    {
        return std::make_unique<WlsGeneratorState<MemSpace::host>>(
            id, capacity, size);
    }
    else if (m == MemSpace::device)
    {
        return std::make_unique<WlsGeneratorState<MemSpace::device>>(
            id, capacity, size);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void WlsGeneratorAction::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void WlsGeneratorAction::step(CoreParams const& params,
                              CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical WLS photons from distribution data.
 *
 * The distributions created by tracks in the previous step are stored in the
 * track slots past the compacted buffer. Since the buffer size is only known
 * on device, the buffer operations are launched over an upper bound \em b of
 * the buffer size plus the number of track slots \em n . The buffer grows as
 * needed to at least \em b + 2n , since the next compaction also covers the
 * distributions created during this step.
 */
template<MemSpace M>
void WlsGeneratorAction::step_impl(CoreParams const& params,
                                   CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<WlsGeneratorState<M>>(*state.aux(), this->aux_id());
    auto counters = aux_state.device_counters();
    CELER_ASSERT(aux_state.num_slots == state.size());

    // Compact the buffer and count the new distributions and queued photons
    size_type const bound = update_buffer(aux_state);

    // Grow the buffer if needed: the next compaction also covers the
    // distributions created during this step
    size_type const next_bound = bound + state.size();
    celeritas::detail::reserve_distributions(
        &aux_state.store, bound, next_bound, state.stream_id(), this->label());
    auto& buffer = aux_state.store.ref().distributions;

    // Generate the optical photons from the distribution data. To avoid
    // synchronization, the number of photons and vacancies are checked on
    // device.
    this->generate(params, state);

    // Compact the buffer again to remove stale distributions and free up
    // space to add new distributions during this step
    detail::select_valid(buffer, bound, counters, state.stream_id());

    // Clear everything the next compaction will read past the buffer: this is
    // unspecified after compacting, and uninitialized if the buffer just grew
    detail::clear_tail(buffer, next_bound, counters, state.stream_id());

    // Update the generator and optical core state counters
    this->update_counters(state);

    // Up to one distribution per track slot can be added during this step
    aux_state.grow_bound(state.size());
    aux_state.copy_async();
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photons.
 */
void WlsGeneratorAction::generate(CoreParams const& params,
                                  CoreStateHost& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<WlsGeneratorState<MemSpace::native>>(*state.aux(),
                                                               this->aux_id());

    // Generate optical photons in vacant track slots
    detail::WlsGeneratorExecutor execute{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        wls_ ? wls_->host_ref() : NativeCRef<WavelengthShiftData>{},
        wls2_ ? wls2_->host_ref() : NativeCRef<WavelengthShiftData>{},
        aux_state.store.ref(),
        aux_state.device_counters()};
    launch_action(state.size(), execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void WlsGeneratorAction::generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
