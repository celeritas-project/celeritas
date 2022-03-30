//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackLauncherImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "sim/CoreTrackData.hh"
#include "sim/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Function-like class to launch a "Track"-dependent function from core data.
 *
 * This class should be used primarily by generated kernel functions.
 */
template<class F>
struct TrackLauncher
{
    //!@{
    //! Type aliases
    using ParamsRef
        = celeritas::CoreParamsData<Ownership::const_reference, MemSpace::native>;
    using StateRef
        = celeritas::CoreStateData<Ownership::reference, MemSpace::native>;
    //!@}

    //// DATA ////

    ParamsRef const& params;
    StateRef const&  states;
    F                call_with_track;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const
    {
        CELER_ASSERT(thread < this->states.size());
        const celeritas::CoreTrackView track(
            this->params, this->states, thread);
        this->call_with_track(track);
    }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
