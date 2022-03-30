//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
    failed,      //!< Recoverable error in sampling
    spawned,     //!< Primary particle from an event or creation of a secondary
    scattered,   //!< Scattering interaction
    msc_limited, //!< Multiple scattering limited process
    entered_volume, //!< Propagated to a new region of space
    unchanged,      //!< Edge cases where an interactor returns no change
    processed,      //!< State change and secondaries have been processed
    // KILLING ACTIONS BELOW
    begin_killed_,
    absorbed = begin_killed_, //!< Absorbed (killed)
    cutoff_energy,            //!< Below energy cutoff (killed)
    escaped,                  //!< Exited geometry (killed)
    end_killed_
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
    CELER_EXPECT(int(a) < int(Action::end_killed_));
    return int(a) >= int(Action::begin_killed_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the given action does not alter the particle's state.
 */
inline CELER_FUNCTION bool action_unchanged(Action a)
{
    return a == Action::unchanged || a == Action::entered_volume
           || a == Action::spawned || a == Action::processed;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the given action returning the msc limited process
 */
inline CELER_FUNCTION bool action_msc(Action a)
{
    return a == Action::msc_limited;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
