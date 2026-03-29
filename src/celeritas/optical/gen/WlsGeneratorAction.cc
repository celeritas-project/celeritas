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
//! Construct a state
template<MemSpace M>
auto make_state(StreamId stream, size_type size)
{
    using StoreT = StateDataStore<WlsGeneratorStateData, M>;

    auto result = std::make_unique<WlsGeneratorState<M>>();
    result->store = StoreT{stream, size};

    CELER_ENSURE(*result);
    return result;
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
                    "optical-wls-generate",
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
        return make_state<MemSpace::host>(id, capacity_);
    }
    else if (m == MemSpace::device)
    {
        return make_state<MemSpace::device>(id, capacity_);
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
 * \todo Accumulate the total number of steps (distributions) in \c
 * accum.buffer_size
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

    auto num_pending_prev = counters.num_pending;

    // Compact the buffer, returning the total number of valid distributions
    counters.buffer_size = celeritas::detail::remove_if_invalid(
        buffer, 0, counters.buffer_size + state.size(), state.stream_id());

    if (counters.buffer_size > 0)
    {
        // If this process created photons, calculate the cumulative sum of the
        // number of photons in the buffered distributions. This is used to
        // determine which thread will generate photons from which distribution
        counters.num_pending = detail::inclusive_scan_photons(
            aux_state.store.ref().distributions,
            aux_state.store.ref().offsets,
            counters.buffer_size,
            state.stream_id());
    }

    // Update the core state counters with the number of new pending tracks
    auto core_counters = state.sync_get_counters();
    core_counters.num_pending += counters.num_pending - num_pending_prev;
    state.sync_put_counters(core_counters);

    if (counters.num_pending > 0 && core_counters.num_vacancies > 0)
    {
        // Generate the optical photons from the distribution data
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
                   << ") for buffered optical WLS distribution data "
                      "(total capacity requirement of "
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
    size_type num_gen = min(state.sync_get_counters().num_vacancies,
                            aux_state.counters.num_pending);

    // Generate optical photons in vacant track slots
    detail::WlsGeneratorExecutor execute{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        wls_ ? wls_->host_ref() : NativeCRef<WavelengthShiftData>{},
        wls2_ ? wls2_->host_ref() : NativeCRef<WavelengthShiftData>{},
        aux_state.store.ref(),
        aux_state.counters.buffer_size};
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
