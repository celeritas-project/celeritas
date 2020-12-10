//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimStatePointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simulation state of a track.
 */
struct SimTrackState
{
    TrackId track_id;      //!< Unique ID for this track
    TrackId parent_id;     //!< ID of parent that created it
    EventId event_id;      //!< ID of originating event
    bool    alive = false; //!< Whether this track is alive
};

//---------------------------------------------------------------------------//
/*!
 * View to the simulation states of multiple tracks.
 */
struct SimStatePointers
{
    span<SimTrackState> vars;

    //! Check whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const { return !vars.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return vars.size(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
