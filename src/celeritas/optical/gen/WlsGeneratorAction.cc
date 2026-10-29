//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/WlsGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "WlsGeneratorAction.hh"

#include <algorithm>

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

#include "GeneratorAction.hh"
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
 * Distributions created during the previous step were written after the end
 * of the buffer, at most one per track slot. This compacts them into the
 * buffer, computes the cumulative photon counts, and accumulates the new
 * distributions and photons.
 */
template<MemSpace M>
void update_buffer(
    WlsGeneratorState<M>& state, StreamId stream_id, size_type num_track_slots)
{
    auto& counters = state.counters;

    // Distributions and photons remaining from the previous step
    auto const prev_buffer_size = counters.buffer_size;
    auto const prev_pending = counters.num_pending;

    // Compact the buffer, returning the total number of valid distributions
    counters.buffer_size = celeritas::detail::remove_if_invalid(
        state.store.ref().distributions,
        0,
        counters.buffer_size + num_track_slots,
        stream_id);

    if (counters.buffer_size > 0)
    {
        // If this process created photons, calculate the cumulative sum of the
        // number of photons in the buffered distributions. This is used to
        // determine which thread will generate photons from which
        // distribution. The new photons were already added to the core
        // state's pending count when the distributions were created.
        counters.num_pending
            = detail::inclusive_scan_photons(state.store.ref().distributions,
                                             state.store.ref().offsets,
                                             counters.buffer_size,
                                             stream_id);
    }

    // Accumulate distributions and photons created during the previous step
    CELER_ASSERT(counters.buffer_size >= prev_buffer_size);
    CELER_ASSERT(counters.num_pending >= prev_pending);
    state.accum.buffer_size += counters.buffer_size - prev_buffer_size;
    state.accum.num_pending += counters.num_pending - prev_pending;
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate distributions from the final step and discard all pending work.
 *
 * WLS interactions update the core pending count immediately, but their
 * distributions are normally accumulated at the next generation step. After
 * counting them, the entire buffer is cleared so that the leftover
 * distributions aren't generated in the next transport loop.
 */
template<MemSpace M>
void reset_impl(
    WlsGeneratorState<M>& state, StreamId stream_id, size_type num_track_slots)
{
    update_buffer(state, stream_id, num_track_slots);

    Filler<WlsDistributionData, M> fill{{}, stream_id};
    fill(state.store.ref().distributions[AllItems<WlsDistributionData, M>{}]);
    state.counters = {};
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with IDs, WLS model, and buffer capacity.
 *
 * The second WLS model is optional.
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
auto WlsGeneratorAction::create_state(MemSpace m, StreamId id, size_type) const
    -> UPState
{
    if (m == MemSpace::host)
    {
        return make_state<MemSpace::host, WlsGeneratorState>(id, capacity_);
    }
    else if (m == MemSpace::device)
    {
        return make_state<MemSpace::device, WlsGeneratorState>(id, capacity_);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Count and discard buffered distributions after an aborted loop.
 */
void WlsGeneratorAction::reset(
    AuxStateVec& aux, StreamId stream_id, size_type num_track_slots) const
{
    auto& state = aux.at(this->aux_id());
    if (auto* s = dynamic_cast<WlsGeneratorState<MemSpace::host>*>(&state))
    {
        return reset_impl(*s, stream_id, num_track_slots);
    }
    else if (
        auto* s = dynamic_cast<WlsGeneratorState<MemSpace::device>*>(&state))
    {
        return reset_impl(*s, stream_id, num_track_slots);
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
 */
template<MemSpace M>
void WlsGeneratorAction::step_impl(CoreParams const& params,
                                   CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    using DistId = ItemId<WlsDistributionData>;
    using DistRange = ItemRange<WlsDistributionData>;

    auto& aux_state = get<WlsGeneratorState<M>>(*state.aux(), this->aux_id());
    auto& counters = aux_state.counters;
    auto& buffer = aux_state.store.ref().distributions;

    // Compact the buffer and count the new distributions and queued photons
    update_buffer(aux_state, state.stream_id(), state.size());

    if (counters.num_pending > 0)
    {
        // Generate the optical photons from the distribution data. To avoid
        // synchronization, we defer the check for vacancies.
        this->generate(params, state);

        // Compact the buffer again to remove stale distributions and free up
        // space to add new distributions during this step
        counters.buffer_size = celeritas::detail::remove_if_invalid(
            buffer, 0, counters.buffer_size, state.stream_id());
    }

    // Ensure the buffer is large enough to hold WLS distributions created
    // during this step
    CELER_VALIDATE(counters.buffer_size + state.size() <= buffer.size(),
                   << "insufficient capacity (" << buffer.size()
                   << ") for buffered optical WLS distribution data (total "
                      "capacity requirement of "
                   << counters.buffer_size + state.size() << ")");

    // Clear data that tracks might write distributions to in this step, which
    // is in an unspecified state after calling \c remove_if on the buffer
    Filler<WlsDistributionData, M> fill{{}, state.stream_id()};
    fill(buffer[DistRange(DistId(counters.buffer_size),
                          DistId(counters.buffer_size + state.size()))]);

    // Update the generator and optical core state counters
    this->update_counters(state);

    CELER_ENSURE(!counters.buffer_size == !counters.num_pending);
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
        aux_state.counters.buffer_size};
    size_type num_gen = std::min(aux_state.counters.num_pending, state.size());
    launch_action(num_gen, execute);
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
