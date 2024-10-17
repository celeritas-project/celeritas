//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SimTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"

#include "SimData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Simulation properties for a single track.
 */
class SimTrackView
{
  public:
    //! Data for initializing the simulation state
    struct Initializer
    {
        real_type time{};
    };

  public:
    // Construct from local data
    inline CELER_FUNCTION
    SimTrackView(NativeRef<SimStateData> const&, TrackSlotId);

    // Initialize the sim state
    inline CELER_FUNCTION SimTrackView& operator=(Initializer const&);

    // Set whether the track is alive
    inline CELER_FUNCTION void status(TrackStatus);

    // Add the time change over the step
    inline CELER_FUNCTION void add_time(real_type delta);

    // Reset step limiter
    inline CELER_FUNCTION void reset_step_limit();

    // Reset step limiter to the given limit
    inline CELER_FUNCTION void reset_step_limit(StepLimit const& sl);

    // Limit the step by this distance and action
    inline CELER_FUNCTION bool step_limit(StepLimit const& sl);

    // Update limiting step
    inline CELER_FUNCTION void step_length(real_type length);

    // Force the limiting action to take
    inline CELER_FUNCTION void post_step_action(ActionId action);

    //// DYNAMIC PROPERTIES ////

    // Time elapsed in the lab frame since the start of the event
    inline CELER_FUNCTION real_type time() const;

    // Whether the track is alive or inactive or dying
    inline CELER_FUNCTION TrackStatus status() const;

    // Limiting step
    inline CELER_FUNCTION real_type step_length() const;

    // Access post-step action to take
    inline CELER_FUNCTION ActionId post_step_action() const;

  private:
    NativeRef<SimStateData> const& states_;
    TrackSlotId track_slot_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from local data.
 */
CELER_FUNCTION
SimTrackView::SimTrackView(NativeRef<SimStateData> const& states,
                           TrackSlotId tid)
    : states_(states), track_slot_(tid)
{
    CELER_EXPECT(track_slot_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the simulation state.
 */
CELER_FUNCTION SimTrackView& SimTrackView::operator=(Initializer const& init)
{
    states_.time[track_slot_] = init.time;
    states_.step_length[track_slot_] = {};
    states_.status[track_slot_] = TrackStatus::initializing;
    states_.post_step_action[track_slot_] = {};
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Set whether the track is active, dying, or inactive.
 */
CELER_FUNCTION void SimTrackView::status(TrackStatus status)
{
    CELER_EXPECT(status != TrackStatus::size_);
    states_.status[track_slot_] = status;
}

//---------------------------------------------------------------------------//
/*!
 * Add the time change over the step.
 */
CELER_FUNCTION void SimTrackView::add_time(real_type delta)
{
    CELER_EXPECT(delta >= 0);
    states_.time[track_slot_] += delta;
}

//---------------------------------------------------------------------------//
/*!
 * Reset step limiter at the beginning of a step.
 *
 * The action can be unset if and only if the step is infinite.
 */
CELER_FUNCTION void SimTrackView::reset_step_limit(StepLimit const& sl)
{
    CELER_EXPECT(sl.step >= 0);
    CELER_EXPECT(static_cast<bool>(sl.action)
                 != (sl.step == numeric_limits<real_type>::infinity()));
    states_.step_length[track_slot_] = sl.step;
    states_.post_step_action[track_slot_] = sl.action;
}

//---------------------------------------------------------------------------//
/*!
 * Reset step limiter at the beginning of a step.
 */
CELER_FUNCTION void SimTrackView::reset_step_limit()
{
    StepLimit limit;
    limit.step = numeric_limits<real_type>::infinity();
    limit.action = {};
    this->reset_step_limit(limit);
}

//---------------------------------------------------------------------------//
/*!
 * Force the limiting action to take.
 *
 * This is used by intermediate kernels (such as \c discrete_select_track )
 * that dispatch to another kernel action before the end of the step without
 * changing the step itself.
 */
CELER_FUNCTION void SimTrackView::post_step_action(ActionId action)
{
    CELER_ASSERT(action);
    states_.post_step_action[track_slot_] = action;
}

//---------------------------------------------------------------------------//
/*!
 * Update the current limiting step.
 */
CELER_FUNCTION void SimTrackView::step_length(real_type length)
{
    CELER_EXPECT(length > 0);
    states_.step_length[track_slot_] = length;
}

//---------------------------------------------------------------------------//
/*!
 * Limit the step by this distance and action.
 *
 * If the step limits are the same, the original action is retained.
 *
 * \return Whether the given limit is the new limit.
 */
CELER_FUNCTION bool SimTrackView::step_limit(StepLimit const& sl)
{
    CELER_ASSERT(sl.step >= 0);

    bool is_limiting = (sl.step < states_.step_length[track_slot_]);
    if (is_limiting)
    {
        states_.step_length[track_slot_] = sl.step;
        states_.post_step_action[track_slot_] = sl.action;
    }
    return is_limiting;
}

//---------------------------------------------------------------------------//
// DYNAMIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Time elapsed in the lab frame since the start of the event [s].
 */
CELER_FORCEINLINE_FUNCTION real_type SimTrackView::time() const
{
    return states_.time[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is inactive, alive, or being killed.
 */
CELER_FORCEINLINE_FUNCTION TrackStatus SimTrackView::status() const
{
    return states_.status[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the current limiting step.
 */
CELER_FORCEINLINE_FUNCTION real_type SimTrackView::step_length() const
{
    return states_.step_length[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Access post-step action to take.
 */
CELER_FORCEINLINE_FUNCTION ActionId SimTrackView::post_step_action() const
{
    return states_.post_step_action[track_slot_];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
