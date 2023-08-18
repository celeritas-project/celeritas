//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/Secondary.hh"

#include "SimData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simulation properties for a single track.
 */
class SimTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using SimParamsRef = NativeCRef<SimParamsData>;
    using SimStateRef = NativeRef<SimStateData>;
    using Initializer_t = SimTrackInitializer;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with view to state and persistent data
    inline CELER_FUNCTION SimTrackView(SimParamsRef const& params,
                                       SimStateRef const& states,
                                       TrackSlotId tid);

    // Initialize the sim state
    inline CELER_FUNCTION SimTrackView& operator=(Initializer_t const& other);

    // Add the time change over the step
    inline CELER_FUNCTION void add_time(real_type delta);

    // Increment the total number of steps
    CELER_FORCEINLINE_FUNCTION void increment_num_steps();

    // Update the number of steps this track has been looping
    inline CELER_FUNCTION void update_looping(bool);

    // Whether the looping track should be abandoned
    inline CELER_FUNCTION bool is_looping(ParticleId, Energy);

    // Set whether the track is alive
    inline CELER_FUNCTION void status(TrackStatus);

    // Reset step limiter
    inline CELER_FUNCTION void reset_step_limit();

    // Reset step limiter to the given limit
    inline CELER_FUNCTION void reset_step_limit(StepLimit const& sl);

    // Limit the step and override if the step is equal
    inline CELER_FUNCTION void force_step_limit(StepLimit const& sl);

    // Limit the step by this distance and action
    inline CELER_FUNCTION bool step_limit(StepLimit const& sl);

    //// DYNAMIC PROPERTIES ////

    // Unique track identifier
    CELER_FORCEINLINE_FUNCTION TrackId track_id() const;

    // Track ID of parent
    CELER_FORCEINLINE_FUNCTION TrackId parent_id() const;

    // Event ID
    CELER_FORCEINLINE_FUNCTION EventId event_id() const;

    // Total number of steps taken by the track
    CELER_FORCEINLINE_FUNCTION size_type num_steps() const;

    // Number of steps taken by the track since it was flagged as looping
    CELER_FORCEINLINE_FUNCTION size_type num_looping_steps() const;

    // Time elapsed in the lab frame since the start of the event [s]
    CELER_FORCEINLINE_FUNCTION real_type time() const;

    // Whether the track is alive or inactive or dying
    CELER_FORCEINLINE_FUNCTION TrackStatus status() const;

    // Limiting step
    CELER_FORCEINLINE_FUNCTION real_type step_length() const;

    // Update limiting step
    CELER_FORCEINLINE_FUNCTION void step_length(real_type length);

    // Access post-step action to take
    inline CELER_FUNCTION ActionId post_step_action() const;

    // Force the limiting action to take
    inline CELER_FUNCTION void post_step_action(ActionId action);

    // Access along-step action to take
    inline CELER_FUNCTION ActionId along_step_action() const;

    // Update along-step action to take
    inline CELER_FUNCTION void along_step_action(ActionId action);

    //// PARAMETER DATA ////

    // Particle-dependent parameters for killing looping tracks
    CELER_FORCEINLINE_FUNCTION
    LoopingThreshold const& looping_threshold(ParticleId) const;

  private:
    SimParamsRef const& params_;
    SimStateRef const& states_;
    const TrackSlotId track_slot_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and local data.
 */
CELER_FUNCTION
SimTrackView::SimTrackView(SimParamsRef const& params,
                           SimStateRef const& states,
                           TrackSlotId tid)
    : params_(params), states_(states), track_slot_(tid)
{
    CELER_EXPECT(track_slot_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Initialize the particle.
 */
CELER_FUNCTION SimTrackView& SimTrackView::operator=(Initializer_t const& other)
{
    states_.track_ids[track_slot_] = other.track_id;
    states_.parent_ids[track_slot_] = other.parent_id;
    states_.event_ids[track_slot_] = other.event_id;
    states_.num_steps[track_slot_] = 0;
    states_.num_looping_steps[track_slot_] = 0;
    states_.time[track_slot_] = other.time;
    states_.status[track_slot_] = other.status;
    states_.steps[track_slot_] = {};
    states_.post_step_action[track_slot_] = {};
    states_.along_step_action[track_slot_] = {};
    return *this;
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
 * Increment the total number of steps.
 */
CELER_FUNCTION void SimTrackView::increment_num_steps()
{
    ++states_.num_steps[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Update the number of steps this track has been looping.
 */
CELER_FUNCTION void SimTrackView::update_looping(bool is_looping)
{
    if (is_looping)
    {
        ++states_.num_looping_steps[track_slot_];
    }
    else
    {
        states_.num_looping_steps[track_slot_] = 0;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Whether the looping track should be abandoned.
 */
CELER_FUNCTION bool SimTrackView::is_looping(ParticleId pid, Energy energy)
{
    auto const& looping = this->looping_threshold(pid);
    if (energy < looping.threshold_energy)
    {
        return this->num_looping_steps() >= looping.max_subthreshold_steps;
    }
    else
    {
        return this->num_looping_steps() >= looping.max_steps;
    }
    return false;
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
    states_.steps[track_slot_] = sl.step;
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
    this->along_step_action({});
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
 * Forcibly limit the step by this distance and action.
 *
 * If the step limits are the same, the new action overrides. The new step must
 * not be greater than the current step.
 */
CELER_FUNCTION void SimTrackView::force_step_limit(StepLimit const& sl)
{
    CELER_ASSERT(sl.step >= 0 && sl.step <= states_.steps[track_slot_]);

    states_.steps[track_slot_] = sl.step;
    states_.post_step_action[track_slot_] = sl.action;
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

    bool is_limiting = (sl.step < states_.steps[track_slot_]);
    if (is_limiting)
    {
        states_.steps[track_slot_] = sl.step;
        states_.post_step_action[track_slot_] = sl.action;
    }
    return is_limiting;
}

//---------------------------------------------------------------------------//
/*!
 * Access post-step action to take.
 */
CELER_FUNCTION ActionId SimTrackView::post_step_action() const
{
    return states_.post_step_action[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Access along-step action to take.
 */
CELER_FUNCTION ActionId SimTrackView::along_step_action() const
{
    return states_.along_step_action[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Update along-step action to take.
 */
CELER_FUNCTION void SimTrackView::along_step_action(ActionId action)
{
    states_.along_step_action[track_slot_] = action;
}

//---------------------------------------------------------------------------//
/*!
 * Set whether the track is active, dying, or inactive.
 */
CELER_FUNCTION void SimTrackView::status(TrackStatus status)
{
    CELER_EXPECT(status != this->status());
    states_.status[track_slot_] = status;
}

//---------------------------------------------------------------------------//
// DYNAMIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Unique track identifier.
 */
CELER_FUNCTION TrackId SimTrackView::track_id() const
{
    return states_.track_ids[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Track ID of parent.
 */
CELER_FUNCTION TrackId SimTrackView::parent_id() const
{
    return states_.parent_ids[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Event ID.
 */
CELER_FUNCTION EventId SimTrackView::event_id() const
{
    return states_.event_ids[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Total number of steps taken by the track.
 */
CELER_FUNCTION size_type SimTrackView::num_steps() const
{
    return states_.num_steps[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Number of steps taken by the track since it was flagged as looping.
 */
CELER_FUNCTION size_type SimTrackView::num_looping_steps() const
{
    return states_.num_looping_steps[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Time elapsed in the lab frame since the start of the event [s].
 */
CELER_FUNCTION real_type SimTrackView::time() const
{
    return states_.time[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is inactive, alive, or being killed.
 */
CELER_FUNCTION TrackStatus SimTrackView::status() const
{
    return states_.status[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the current limiting step.
 */
CELER_FUNCTION real_type SimTrackView::step_length() const
{
    return states_.steps[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Update the current limiting step.
 */
CELER_FUNCTION void SimTrackView::step_length(real_type length)
{
    CELER_EXPECT(length > 0);
    states_.steps[track_slot_] = length;
}

//---------------------------------------------------------------------------//
/*!
 * Particle-dependent parameters for killing looping tracks.
 */
CELER_FORCEINLINE_FUNCTION LoopingThreshold const&
SimTrackView::looping_threshold(ParticleId pid) const
{
    return params_.looping[pid];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
