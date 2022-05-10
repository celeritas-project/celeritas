//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
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
    //! Type aliases
    using SimStateRef   = SimStateData<Ownership::reference, MemSpace::native>;
    using Initializer_t = SimTrackInitializer;
    //!@}

  public:
    // Construct with view to state and persistent data
    inline CELER_FUNCTION
    SimTrackView(const SimStateRef& states, ThreadId thread);

    // Initialize the sim state
    inline CELER_FUNCTION SimTrackView& operator=(const Initializer_t& other);

    // Increment the total number of steps
    CELER_FORCEINLINE_FUNCTION void increment_num_steps();

    // Set whether the track is alive
    inline CELER_FUNCTION void status(TrackStatus);

    // Reset step limiter
    inline CELER_FUNCTION void reset_step_limit();

    // Reset step limiter to the given limit
    inline CELER_FUNCTION void reset_step_limit(const StepLimit& sl);

    // Force the limiting action to take
    inline CELER_FUNCTION void force_step_limit(ActionId action);

    // Limit the step and override if the step is equal
    inline CELER_FUNCTION void force_step_limit(const StepLimit& sl);

    // Limit the step by this distance and action
    inline CELER_FUNCTION bool step_limit(const StepLimit& sl);

    //// DYNAMIC PROPERTIES ////

    // Unique track identifier
    CELER_FORCEINLINE_FUNCTION TrackId track_id() const;

    // Track ID of parent
    CELER_FORCEINLINE_FUNCTION TrackId parent_id() const;

    // Event ID
    CELER_FORCEINLINE_FUNCTION EventId event_id() const;

    // Total number of steps taken by the track
    CELER_FORCEINLINE_FUNCTION size_type num_steps() const;

    // Whether the track is alive or inactive or dying
    CELER_FORCEINLINE_FUNCTION TrackStatus status() const;

    // Limiting step and action to take
    CELER_FORCEINLINE_FUNCTION const StepLimit& step_limit() const;

  private:
    const SimStateRef& states_;
    const ThreadId     thread_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and local data.
 */
CELER_FUNCTION
SimTrackView::SimTrackView(const SimStateRef& states, ThreadId thread)
    : states_(states), thread_(thread)
{
    CELER_EXPECT(thread < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Initialize the particle.
 */
CELER_FUNCTION SimTrackView& SimTrackView::operator=(const Initializer_t& other)
{
    states_.state[thread_] = other;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Increment the total number of steps.
 */
CELER_FUNCTION void SimTrackView::increment_num_steps()
{
    ++states_.state[thread_].num_steps;
}

//---------------------------------------------------------------------------//
/*!
 * Reset step limiter at the beginning of a step.
 *
 * The action can be unset if and only if the step is infinite.
 */
CELER_FUNCTION void SimTrackView::reset_step_limit(const StepLimit& sl)
{
    CELER_EXPECT(sl.step >= 0);
    CELER_EXPECT(static_cast<bool>(sl.action)
                 != (sl.step == numeric_limits<real_type>::infinity()));
    states_.state[thread_].step_limit = sl;
}

//---------------------------------------------------------------------------//
/*!
 * Reset step limiter at the beginning of a step.
 */
CELER_FUNCTION void SimTrackView::reset_step_limit()
{
    StepLimit limit;
    limit.step   = numeric_limits<real_type>::infinity();
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
CELER_FUNCTION void SimTrackView::force_step_limit(ActionId action)
{
    CELER_ASSERT(action);
    states_.state[thread_].step_limit.action = action;
}

//---------------------------------------------------------------------------//
/*!
 * Forcibly limit the step by this distance and action.
 *
 * If the step limits are the same, the new action overrides. The new step must
 * not be greater than the current step.
 */
CELER_FUNCTION void SimTrackView::force_step_limit(const StepLimit& sl)
{
    CELER_ASSERT(sl.step >= 0
                 && sl.step <= states_.state[thread_].step_limit.step);

    states_.state[thread_].step_limit = sl;
}

//---------------------------------------------------------------------------//
/*!
 * Limit the step by this distance and action.
 *
 * If the step limits are the same, the original action is retained.
 *
 * \return Whether the given limit is the new limit.
 */
CELER_FUNCTION bool SimTrackView::step_limit(const StepLimit& sl)
{
    CELER_ASSERT(sl.step >= 0);

    bool is_limiting = (sl.step < states_.state[thread_].step_limit.step);
    if (is_limiting)
    {
        states_.state[thread_].step_limit = sl;
    }
    return is_limiting;
}

//---------------------------------------------------------------------------//
/*!
 * Set whether the track is active, dying, or inactive.
 */
CELER_FUNCTION void SimTrackView::status(TrackStatus status)
{
    CELER_EXPECT(status != this->status());
    states_.state[thread_].status = status;
}

//---------------------------------------------------------------------------//
// DYNAMIC PROPERTIES
//---------------------------------------------------------------------------//
/*!
 * Unique track identifier.
 */
CELER_FUNCTION TrackId SimTrackView::track_id() const
{
    return states_.state[thread_].track_id;
}

//---------------------------------------------------------------------------//
/*!
 * Track ID of parent.
 */
CELER_FUNCTION TrackId SimTrackView::parent_id() const
{
    return states_.state[thread_].parent_id;
}

//---------------------------------------------------------------------------//
/*!
 * Event ID.
 */
CELER_FUNCTION EventId SimTrackView::event_id() const
{
    return states_.state[thread_].event_id;
}

//---------------------------------------------------------------------------//
/*!
 * Total number of steps taken by the track.
 */
CELER_FUNCTION size_type SimTrackView::num_steps() const
{
    return states_.state[thread_].num_steps;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is inactive, alive, or being killed.
 */
CELER_FUNCTION TrackStatus SimTrackView::status() const
{
    return states_.state[thread_].status;
}

//---------------------------------------------------------------------------//
/*!
 * Get the current limiting step and action.
 */
CELER_FUNCTION const StepLimit& SimTrackView::step_limit() const
{
    return states_.state[thread_].step_limit;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
