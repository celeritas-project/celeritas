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

    //// DYNAMIC PROPERTIES ////

    // Unique track identifier
    CELER_FORCEINLINE_FUNCTION TrackId track_id() const;

    // Track ID of parent
    CELER_FORCEINLINE_FUNCTION TrackId parent_id() const;

    // Event ID
    CELER_FORCEINLINE_FUNCTION EventId event_id() const;

    // Whether the track is alive
    CELER_FORCEINLINE_FUNCTION bool alive() const;

    // Set whether the track is alive
    CELER_FORCEINLINE_FUNCTION void alive(bool);

  private:
    const SimStateRef& states_;
    const ThreadId     thread_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "SimTrackView.i.hh"
