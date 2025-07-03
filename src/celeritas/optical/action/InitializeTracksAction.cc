//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/InitializeTracksAction.cc
//---------------------------------------------------------------------------//
#include "InitializeTracksAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/TrackInitParams.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/InitTracksExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
InitializeTracksAction::InitializeTracksAction(ActionId aid)
    : ConcreteAction(aid, "initialize-tracks", "initialize track states")
{
}

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
 * Initialize optical track states.
 */
template<MemSpace M>
void InitializeTracksAction::step_impl(CoreParams const& params,
                                       CoreState<M>& state) const
{
    auto& counters = state.counters();

    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers
    size_type num_new_tracks
        = std::min(counters.num_vacancies, counters.num_initializers);
    if (num_new_tracks > 0)
    {
        //! \todo Execute this block if warming up

        // Launch a kernel to initialize tracks
        this->step_impl(params, state, num_new_tracks);

        // Update initializers/vacancies
        counters.num_initializers -= num_new_tracks;
        counters.num_vacancies -= num_new_tracks;
    }

    // Store number of active tracks at the start of the loop
    counters.num_active = state.size() - counters.num_vacancies;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to initialize tracks.
 *
 * The thread index here corresponds to initializer indices, not track slots
 * (or indicies into the track slot indirection array).
 */
void InitializeTracksAction::step_impl(CoreParams const& params,
                                       CoreStateHost& state,
                                       size_type num_new_tracks) const
{
    detail::InitTracksExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), state.counters()};
    return launch_action(num_new_tracks, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void InitializeTracksAction::step_impl(CoreParams const&,
                                       CoreStateDevice&,
                                       size_type) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
