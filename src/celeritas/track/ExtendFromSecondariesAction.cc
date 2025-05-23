//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.cc
//---------------------------------------------------------------------------//
#include "ExtendFromSecondariesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/LocateAliveExecutor.hh"  // IWYU pragma: associated
#include "detail/ProcessSecondariesExecutor.hh"  // IWYU pragma: associated
#include "detail/TrackInitAlgorithms.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a long description of the action.
 */
std::string_view ExtendFromSecondariesAction::description() const
{
    return "create track initializers from secondaries";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void ExtendFromSecondariesAction::step(CoreParams const& params,
                                       CoreStateHost& state) const
{
    return this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void ExtendFromSecondariesAction::step(CoreParams const& params,
                                       CoreStateDevice& state) const
{
    return this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track states.
 */
template<MemSpace M>
void ExtendFromSecondariesAction::step_impl(CoreParams const& core_params,
                                            CoreState<M>& core_state) const
{
    TrackInitStateData<Ownership::reference, M>& init = core_state.ref().init;
    CoreStateCounters& counters = core_state.counters();

    // Launch a kernel to identify which track slots are still alive and count
    // the number of surviving secondaries per track
    this->locate_alive(core_params, core_state);

    // Remove all elements in the vacancy vector that were flagged as active
    // tracks, leaving the (sorted) indices of the empty slots
    counters.num_vacancies
        = detail::remove_if_alive(init.vacancies, core_state.stream_id());

    // The exclusive prefix sum of the number of secondaries produced by each
    // track is used to get the start index in the vector of track initializers
    // for each thread. Starting at that index, each thread creates track
    // initializers from all surviving secondaries produced in its
    // interaction.
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
     * - Copy to device to update the on-device refrences (state.ptr)
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
    this->process_secondaries(core_params, core_state);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to locate alive particles.
 *
 * This fills the TrackInit \c vacancies and \c secondary_counts arrays.
 */
void ExtendFromSecondariesAction::locate_alive(CoreParams const& core_params,
                                               CoreStateHost& core_state) const
{
    detail::LocateAliveExecutor execute{core_params.ptr<MemSpace::native>(),
                                        core_state.ptr()};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondary particles.
 */
void ExtendFromSecondariesAction::process_secondaries(
    CoreParams const& core_params, CoreStateHost& core_state) const
{
    //! \todo Wrap with a regular track executor but without remapping slots?
    detail::ProcessSecondariesExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        core_state.counters()};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
// DEVICE-DISABLED IMPLEMENTATION
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ExtendFromSecondariesAction::begin_run(CoreParams const&, CoreStateDevice&)
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void ExtendFromSecondariesAction::locate_alive(CoreParams const&,
                                               CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

void ExtendFromSecondariesAction::process_secondaries(CoreParams const&,
                                                      CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
