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
template<class F, class... Ts>
struct TrackLauncherImpl;

//---------------------------------------------------------------------------//
template<class F>
struct TrackLauncherImpl<F>
{
    F call_with_track;

    CELER_FUNCTION void operator()(CoreTrackView const& track) const
    {
        return this->call_with_track(track);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
