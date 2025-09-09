//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/StatusCheckExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../SimTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#    include "celeritas/global/Debug.hh"
#endif

#include "../StatusCheckData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
#if !CELER_DEVICE_COMPILE
// Print a track and why it failed.
#    define CELER_PRINT_TRACK(MSG, TRACK) \
        CELER_LOG_LOCAL(error)            \
            << MSG << ": " << ::celeritas::StreamableTrack{TRACK};
#else
#    define CELER_PRINT_TRACK(MSG, TRACK)
#endif

/*!
 * Check that the condition is true, otherwise throw an error/assertion.
 *
 * \note This macro is defined so that the condition is still checked in
 * "release" mode, and so that checking will work on GPU.
 */
#define CELER_FAIL_IF(COND, MSG)                               \
    do                                                         \
    {                                                          \
        if (CELER_UNLIKELY((COND)))                            \
        {                                                      \
            CELER_PRINT_TRACK(MSG, track)                      \
            CELER_DEBUG_THROW_(MSG ": '" #COND "'", internal); \
        }                                                      \
    } while (0)

//---------------------------------------------------------------------------//
/*!
 * Assert that a track's status and actions are valid.
 *
 * When enabled as a debug option, this is called after every action is
 * executed. The state's "action" and "order" are for the last executed action.
 * The status, post-step, and along-step action are from *before* the last
 * executed action.
 *
 * See \c StepActionOrder, \c TrackStatus .
 */
struct StatusCheckExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    NativeCRef<StatusCheckParamsData> const params;
    NativeRef<StatusCheckStateData> const state;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION void StatusCheckExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state.order != StepActionOrder::size_);
    CELER_EXPECT(state.action);

    auto tsid = track.track_slot_id();
    auto sim = track.sim();

    if (state.order > StepActionOrder::start
        && state.order < StepActionOrder::end)
    {
        auto prev_status = state.status[tsid];
        CELER_FAIL_IF(sim.status() < prev_status,
                      "status was improperly reverted");
    }
    if (state.order >= StepActionOrder::pre
        && state.order < StepActionOrder::end)
    {
        // Initializing takes place *either* at the very beginning of the
        // stepping loop *or* at the very end (in the case where a track is
        // initialized in-place from a secondary). It should be cleared in
        // pre-step
        CELER_FAIL_IF(sim.status() == TrackStatus::initializing,
                      "status cannot be 'initializing' after pre-step");
    }
    if (sim.status() == TrackStatus::inactive)
    {
        // Remaining tests only apply to active tracks
        return;
    }
    if (state.order < StepActionOrder::pre
        || state.order == StepActionOrder::end)
    {
        // Skip remaining tests since step actions get reset in "pre-step"
        return;
    }

    if (sim.step_length() != numeric_limits<real_type>::infinity())
    {
        /*!
         * It's allowable to have *no* post step action if there are no physics
         * processes for the current particle type.
         * \todo Change this behavior to be a *tracking cut* rather than lost
         * energy.
         */
        CELER_FAIL_IF(!sim.post_step_action(), "missing post-step action");
    }

    if (sim.status() == TrackStatus::alive)
    {
        // If the track fails during initialization, it won't get an
        // along-step action, so only check this if alive
        CELER_FAIL_IF(!sim.along_step_action(), "missing along-step action");

        // All 'alive' tracks should be inside the geometry
        CELER_FAIL_IF(track.geometry().is_outside(),
                      "track is outside the geometry but still 'alive'");
    }

    ActionId const prev_along_step = state.along_step_action[tsid];
    ActionId const next_along_step = sim.along_step_action();
    if (state.order > StepActionOrder::pre && next_along_step)
    {
        CELER_FAIL_IF(prev_along_step != next_along_step,
                      "along-step action cannot yet change");
    }

    ActionId const prev_post_step = state.post_step_action[tsid];
    ActionId const next_post_step = sim.post_step_action();
    if (state.order > StepActionOrder::pre && prev_post_step
        && prev_post_step != next_post_step)
    {
        // Check that order is increasing if not an "implicit" action
        auto prev_order = params.orders[prev_post_step];
        auto next_order = params.orders[next_post_step];
        CELER_FAIL_IF(!(prev_order == params.implicit_order
                        || next_order == params.implicit_order
                        || OrderedAction{prev_order, prev_post_step}
                               < OrderedAction{next_order, next_post_step}),
                      "new post-step action is out of order");
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
