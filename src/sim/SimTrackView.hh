//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "SimStatePointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simulation properties for a single track.
 *
 * Manage the simulation state.
 */
class SimTrackView
{
  public:
    //@{
    //! Type aliases
    using Initializer_t = SimTrackState;
    //@}

  public:
    // Construct with view to state and persistent data
    inline CELER_FUNCTION
    SimTrackView(const SimStatePointers& state, ThreadId thread);

    // Initialize the sim state
    inline CELER_FUNCTION SimTrackView& operator=(const Initializer_t& other);

    /// DYNAMIC PROPERTIES ///

    //@{
    //! State accessors
    CELER_FUNCTION TrackId track_id() const { return state_.track_id; }
    CELER_FUNCTION TrackId parent_id() const { return state_.parent_id; }
    CELER_FUNCTION EventId event_id() const { return state_.event_id; }
    CELER_FUNCTION bool    alive() const { return state_.alive; }
    //@}

    //@{
    //! State modifiers via non-const references
    CELER_FUNCTION bool& alive() { return state_.alive; }
    //@}

  private:
    SimTrackState& state_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "SimTrackView.i.hh"
