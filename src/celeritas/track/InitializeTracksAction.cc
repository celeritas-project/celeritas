//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/InitializeTracksAction.cc
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/ExecuteAction.hh"

#include "detail/InitTracksLauncher.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void InitializeTracksAction::execute(CoreParams const& params,
                                     CoreStateHost& state) const
{
    return this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void InitializeTracksAction::execute(CoreParams const& params,
                                     CoreStateDevice& state) const
{
    return this->execute_impl(params, state);
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
void InitializeTracksAction::execute_impl(CoreParams const& core_params,
                                          CoreState<M>& core_state) const
{
    auto& scalars = core_state.ref().init.scalars;

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_new_tracks
        = std::min(scalars.num_vacancies, scalars.num_initializers);
    if (num_new_tracks > 0)
    {
        // Launch a kernel to initialize tracks
        this->launch_impl(core_params, core_state, num_new_tracks);

        // Update initializers/vacancies
        scalars.num_initializers -= num_new_tracks;
        scalars.num_vacancies -= num_new_tracks;
    }

    // Store number of active tracks at the start of the loop
    scalars.num_active = core_state.size() - scalars.num_vacancies;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to initialize tracks.
 */
void InitializeTracksAction::launch_impl(CoreParams const& core_params,
                                         CoreStateHost& core_state,
                                         size_type num_new_tracks) const
{
    execute_action(
        *this,
        Range{ThreadId{num_new_tracks}},
        core_params,
        core_state,
        detail::InitTracksLauncher{core_params.ptr<MemSpace::native>(),
                                   core_state.ptr(),
                                   num_new_tracks,
                                   core_state.ref().init.scalars});
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void InitializeTracksAction::launch_impl(CoreParams const&,
                                         CoreStateDevice&,
                                         size_type) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
