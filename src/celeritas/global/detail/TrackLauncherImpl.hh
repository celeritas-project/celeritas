//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/TrackLauncherImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/sys/ThreadId.hh"
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
    //// DATA ////

    NativeCRef<CoreParamsData> const& params;
    NativeRef<CoreStateData> const& state;
    F call_with_track;

    //// METHODS ////

    CELER_FUNCTION void operator()(ThreadId thread) const
    {
        CELER_ASSERT(thread < this->state.size());
        const celeritas::CoreTrackView track(this->params, this->state, thread);
        this->call_with_track(track);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
