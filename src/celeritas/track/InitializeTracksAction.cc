//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.cc
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include "corecel/Macros.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreState.hh"

#include "CounterAlgorithms.hh"
#include "TrackInitParams.hh"  // IWYU pragma: keep

#include "detail/InitTracksExecutor.hh"  // IWYU pragma: associated
#include "detail/TrackInitAlgorithms.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void InitializeTracksAction::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    return this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void InitializeTracksAction::step(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    return this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize track states.
 */
template<MemSpace M>
void InitializeTracksAction::step_impl(CoreParams const& core_params,
                                       CoreState<M>& core_state) const
{
    if (core_params.init()->track_order() == TrackOrder::init_charge)
    {
        // Count neutral tracks to partition the vacancies by charge
        detail::scan_neutral_initializers(
            core_params, core_state.ref().init, core_state.stream_id());
    }

    // Launch a kernel to initialize tracks
    this->step_impl(core_params, core_state, core_state.size());

    // Store number of active tracks at the start of the loop
    update_active(core_state.ref().init.counters,
                  core_state.size(),
                  core_state.stream_id());
}

//---------------------------------------------------------------------------//
/*!
 * Launch (host) kernel to initialize tracks and to update the corresponding
 * counters.
 *
 * The thread index here corresponds to initializer indices, not track slots
 * (or indices into the track slot indirection array).
 */
void InitializeTracksAction::step_impl(CoreParams const& core_params,
                                       CoreStateHost& core_state,
                                       size_type max_new_tracks) const
{
    detail::InitTracksExecutor execute{core_params.ptr<MemSpace::native>(),
                                       core_state.ptr()};
    launch_action(*this, max_new_tracks, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void InitializeTracksAction::step_impl(
    CoreParams const&, CoreStateDevice&, size_type) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
