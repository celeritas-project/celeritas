//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.cu
//---------------------------------------------------------------------------//
#include "ExtendFromSecondariesAction.hh"

#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/LocateAliveExecutor.hh"
#include "detail/ProcessSecondariesExecutor.hh"
#include "detail/TrackInitAlgorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Warm up asynchronous allocation.
 *
 * This just calls MallocAsync before the first step, since it's used by
 * \c detail::remove_if_alive under the hood.
 */
void ExtendFromSecondariesAction::begin_run(CoreParams const&,
                                            CoreStateDevice& core_state)
{
    ScopedProfiling profile_this{this->label()};
    Stream& s = device().stream(core_state.stream_id());
    void* p = s.malloc_async(core_state.size() * sizeof(size_type));
    s.free_async(p);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track states.
 */
void ExtendFromSecondariesAction::step_impl(CoreParams const& core_params,
                                            CoreStateDevice& core_state) const
{
    TrackInitStateData<Ownership::reference, MemSpace::device>& init
        = core_state.ref().init;

    // Launch a kernel to identify which track slots are still alive and count
    // the number of surviving secondaries per track
    this->locate_alive(core_params, core_state);

    // Remove all elements in the vacancy vector that were flagged as active
    // tracks, leaving the (sorted) indices of the empty slots
    detail::remove_if_alive(init, core_state.stream_id());

    // The exclusive prefix sum of the number of secondaries produced by each
    // track is used to get the start index in the vector of track initializers
    // for each thread. Starting at that index, each thread creates track
    // initializers from all surviving secondaries produced in its
    // interaction.
    auto counters = core_state.sync_get_counters();
    counters.num_secondaries = detail::exclusive_scan_counts(
        init.secondary_counts, core_state.stream_id());

    /*! \todo If we don't have space for all the secondaries, we will need to
     * buffer the current track initializers to create room.
     *
     * This isn't trivial because we will need to:
     * - Allocate a new buffer (probably do something like 2x, rounding up to
     *   nearest power of 2)?
     * - Update the collection references for track sim
     * - Update the *copies* of that reference (?) like in track state
     * - Copy to device to update the on-device references (state.ptr)
     */
    counters.num_initializers += counters.num_secondaries;
    CELER_VALIDATE(
        counters.num_initializers <= init.initializers.size(),
        << "insufficient capacity (" << init.initializers.size()
        << ") for track initializers (created " << counters.num_secondaries
        << " new secondaries for a total capacity requirement of "
        << counters.num_initializers
        << "): increase initializer capacity or decrease track slots");

    // Launch a kernel to create track initializers from secondaries
    counters.num_alive = core_state.size() - counters.num_vacancies;
    core_state.sync_put_counters(counters);

    this->process_secondaries(core_params, core_state);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to locate alive particles.
 *
 * This fills the TrackInit \c vacancies and \c secondary_counts arrays.
 */
void ExtendFromSecondariesAction::locate_alive(CoreParams const& core_params,
                                               CoreStateDevice& core_state) const
{
    ScopedProfiling profile_this{"locate-alive"};
    using Executor = detail::LocateAliveExecutor;
    static ActionLauncher<Executor> launch(*this, "locate-alive");
    launch(core_state,
           Executor{core_params.ptr<MemSpace::native>(), core_state.ptr()});
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to create track initializers from secondary particles.
 */
void ExtendFromSecondariesAction::process_secondaries(
    CoreParams const& core_params, CoreStateDevice& core_state) const
{
    ScopedProfiling profile_this{"process-secondaries"};
    using Executor = detail::ProcessSecondariesExecutor;
    static ActionLauncher<Executor> launch(*this, "process-secondaries");
    launch(core_state,
           Executor{core_params.ptr<MemSpace::native>(),
                    core_state.ptr(),
                    core_state.sync_get_counters()});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
