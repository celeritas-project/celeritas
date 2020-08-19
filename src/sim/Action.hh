//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Action.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Broad categories of events applied to a particle in a track.
 *
 * These actions should all read as past-tense verbs, since they are meant
 * to reflect a change in state.
 */
enum class Action
{
    failed,    //!< Recoverable error in sampling
    spawned,   //!< Primary particle from an event or creation of a secondary
    scattered, //!< Scattering interaction
    entered_volume, //!< Propagated to a new region of space
    // KILLING ACTIONS BELOW
    begin_killed_,
    absorbed = begin_killed_, //!< Absorbed (killed)
    cutoff_energy,            //!< Below energy cutoff (killed)
    escaped,                  //!< Exited geometry (killed)
    end_killed_
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the function returning an action succeeded.
 *
 * This should be 'false' for (e.g.) failing to allocate memory for sampling
 * secondaries, allowing a recoverable failure (next kernel launch retries with
 */
inline CELER_FUNCTION bool action_completed(Action a)
{
    return a != Action::failed;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the given action kills the active track.
 */
inline CELER_FUNCTION bool action_killed(Action a)
{
    REQUIRE(int(a) < int(Action::end_killed_));
    return int(a) >= int(Action::begin_killed_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
