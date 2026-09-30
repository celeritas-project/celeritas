//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.cc
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "TrackInitParams.hh"

#include "detail/InitTracksExecutor.hh"  // IWYU pragma: associated
#include "detail/TrackInitAlgorithms.hh"
#include "detail/UpdateNumActiveExecutor.hh"  // IWYU pragma: associated

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
 *
 * Tracks created from secondaries produced in this step will have the geometry
 * state copied over from the parent instead of initialized from the position.
 * If there are more empty slots than new secondaries, they will be filled by
 * any track initializers remaining from previous steps using the position.
 */

template<MemSpace M>
void InitializeTracksAction::step_impl(CoreParams const& core_params,
                                       CoreState<M>& core_state) const
{
    auto counters = core_state.sync_get_counters();

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_new_tracks
        = std::min(counters.num_vacancies, counters.num_initializers);
    if (num_new_tracks > 0)
    {
        if (core_params.init()->track_order() == TrackOrder::init_charge)
        {
            // Reset track initializer indices
            fill_sequence(&core_state.ref().init.indices,
                          core_state.stream_id());

            // Partition indices by whether tracks are charged or neutral
            detail::partition_initializers(core_params,
                                           core_state.ref().init,
                                           num_new_tracks,
                                           core_state.stream_id());
        }

        // Launch a kernel to initialize tracks
        this->step_impl(core_params, core_state, num_new_tracks);
    }

    // Store number of active tracks at the start of the loop
    this->update_num_active(core_params, core_state);
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
                                       size_type num_new_tracks) const
{
    detail::InitTracksExecutor execute{core_params.ptr<MemSpace::native>(),
                                       core_state.ptr()};
    launch_action(*this, num_new_tracks, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Launch (host) kernel to update the corresponding counters.
 *
 */
void InitializeTracksAction::update_num_active(CoreParams const& core_params,
                                               CoreStateHost& core_state) const
{
    // Store number of active tracks at the start of the loop, and update the
    // number of vacancies and initializers if num_new_tracks > 0
    auto execute_thread = make_single_track_executor(
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::UpdateNumActiveExecutor{core_state.size()});
    launch_core(1, "update-active", core_params, core_state, execute_thread);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void InitializeTracksAction::step_impl(
    CoreParams const&, CoreStateDevice&, size_type) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
void InitializeTracksAction::update_num_active(CoreParams const&,
                                               CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
