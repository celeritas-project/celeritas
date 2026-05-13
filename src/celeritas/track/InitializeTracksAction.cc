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
    // The number of new tracks to initialize is the smaller of the number of
    // empty slots in the track vector and the number of track initializers.
    // To avoid synchronizing the core state counters, we let the kernels
    // calculate the number of new tracks and proceed accordingly. This means
    // the code might sometimes call these functions when there is no work
    // to do, but that's quickly determined so the overhead should be minimal.
    if (core_params.init()->track_order() == TrackOrder::init_charge)
    {
        // Reset track initializer indices
        fill_sequence(&core_state.ref().init.indices, core_state.stream_id());

        // Partition indices by whether tracks are charged or neutral
        detail::partition_initializers(
            core_params, core_state.ref().init, core_state.stream_id());
    }

    // Launch a kernel to initialize tracks, using the largest possible
    // number and computing the actual number in the kernel.
    this->step_impl(core_params, core_state, core_state.size());
}

//---------------------------------------------------------------------------//
/*!
 * Launch (host) kernels to initialize tracks and to update the corresponding
 * counters.
 *
 * The thread index here corresponds to initializer indices, not track slots
 * (or indices into the track slot indirection array).
 */
void InitializeTracksAction::step_impl(CoreParams const& core_params,
                                       CoreStateHost& core_state,
                                       size_type max_new_tracks) const
{
    {
        detail::InitTracksExecutor execute{core_params.ptr<MemSpace::native>(),
                                           core_state.ptr()};
        launch_action(*this, max_new_tracks, core_params, core_state, execute);
    }
    {
        detail::UpdateNumActiveExecutor execute_thread{
            core_params.ptr<MemSpace::native>(), core_state.ptr()};
        launch_action(*this, 1, core_params, core_state, execute_thread);
    }
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
}  // namespace celeritas
