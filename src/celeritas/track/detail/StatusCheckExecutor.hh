//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/StatusCheckExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/track/SimTrackView.hh"

#include "../StatusCheckData.hh"

//! Check that the condition is true, otherwise throw an error/assertion
#define SCE_ASSERT(COND, MSG)                                         \
    do                                                                \
    {                                                                 \
        if (CELER_UNLIKELY(!(COND)))                                  \
        {                                                             \
            CELER_DEBUG_THROW_(MSG ": '" #COND "' failed", internal); \
        }                                                             \
    } while (0)

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Assert that a track's status and actions are valid.
 *
 * When enabled as a debug option, this is called after every action is
 * executed. The state's "action" and "order" are for the last executed action.
 * The status, post-step, and along-step action are from *before* the last
 * executed action.
 *
 * See \c ActionOrder, \c TrackStatus .
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
    CELER_EXPECT(state.order != ActionOrder::size_);
    CELER_EXPECT(state.action);

    auto tsid = track.track_slot_id();
    auto sim = track.make_sim_view();

    if (state.order > ActionOrder::start)
    {
        SCE_ASSERT(sim.status() >= state.status[tsid],
                   "status was improperly reverted");
    }
    if (state.order > ActionOrder::user_start)
    {
        SCE_ASSERT(sim.status() != TrackStatus::initializing,
                   "status cannot be 'initializing' after user start");
    }
    if (sim.status() == TrackStatus::inactive)
    {
        // Remaining tests only apply to active tracks
        return;
    }
    if (state.order < ActionOrder::pre)
    {
        // Skip remaining tests since step actions get reset in "pre-step"
        return;
    }

    if (sim.step_length() != numeric_limits<real_type>::infinity())
    {
        // It's allowable to have *no* post step action if there are no physics
        // processes for the current particle type.
        // TODO: change this behavior to be a *tracking cut* rather than
        // lost energy
        SCE_ASSERT(sim.post_step_action(), "missing post-step action");
    }
    SCE_ASSERT(sim.along_step_action(), "missing along-step action");

    ActionId const last_along_step = state.along_step_action[tsid];
    ActionId const next_along_step = sim.along_step_action();
    if (state.order > ActionOrder::pre)
    {
        SCE_ASSERT(last_along_step == next_along_step,
                   "along-step action cannot yet change");
    }

    ActionId const last_post_step = state.post_step_action[tsid];
    ActionId const next_post_step = sim.post_step_action();
    if (state.order > ActionOrder::pre && last_post_step
        && last_post_step != next_post_step)
    {
        // Check that order is increasing if not an "implicit" action
        auto last_order = params.orders[last_post_step];
        auto next_order = params.orders[next_post_step];
        SCE_ASSERT((next_order == params.implicit_order
                    || OrderedAction{last_order, last_post_step}
                           < OrderedAction{next_order, next_post_step}),
                   "new post-step action is out of order");
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
