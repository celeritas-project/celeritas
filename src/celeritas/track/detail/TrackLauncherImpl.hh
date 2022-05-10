//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackLauncherImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"

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
    using CoreRefNative = celeritas::CoreRef<MemSpace::native>;

    //// DATA ////

    CoreRefNative const& data;
    F                    call_with_track;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const
    {
        CELER_ASSERT(thread < this->data.states.size());
        const celeritas::CoreTrackView track(
            this->data.params, this->data.states, thread);
        this->call_with_track(track);
    }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
