//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimTrackView.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
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
 * Whether the track is alive.
 */
CELER_FUNCTION bool SimTrackView::alive() const
{
    return states_.state[thread_].alive;
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
} // namespace celeritas
