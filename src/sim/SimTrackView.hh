//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

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

    // Set the total number of steps
    CELER_FORCEINLINE_FUNCTION void steps(size_type);

    // Set whether the track is alive
    CELER_FORCEINLINE_FUNCTION void alive(bool);

    //// DYNAMIC PROPERTIES ////

    // Unique track identifier
    CELER_FORCEINLINE_FUNCTION TrackId track_id() const;

    // Track ID of parent
    CELER_FORCEINLINE_FUNCTION TrackId parent_id() const;

    // Event ID
    CELER_FORCEINLINE_FUNCTION EventId event_id() const;

    // Total number of steps taken by the track
    CELER_FORCEINLINE_FUNCTION size_type steps() const;

    // Whether the track is alive
    CELER_FORCEINLINE_FUNCTION bool alive() const;

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
 * Set the total number of steps.
 */
CELER_FUNCTION void SimTrackView::steps(size_type count)
{
    states_.state[thread_].steps = count;
}

//---------------------------------------------------------------------------//
/*!
 * Set whether the track is alive.
 */
CELER_FUNCTION void SimTrackView::alive(bool is_alive)
{
    states_.state[thread_].alive = is_alive;
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
CELER_FUNCTION size_type SimTrackView::steps() const
{
    return states_.state[thread_].steps;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is alive.
 */
CELER_FUNCTION bool SimTrackView::alive() const
{
    return states_.state[thread_].alive;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
