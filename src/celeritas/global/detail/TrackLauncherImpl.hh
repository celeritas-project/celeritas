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
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// TrackLauncherImpl
//---------------------------------------------------------------------------//
//! Lazy implementation that replaces tuple + apply
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
template<class F, class T1>
struct TrackLauncherImpl<F, T1>
{
    F call_with_track;
    T1 arg1;

    CELER_FUNCTION void operator()(CoreTrackView const& track) const
    {
        return this->call_with_track(track, arg1);
    }
};

//---------------------------------------------------------------------------//
template<class F, class T1, class T2>
struct TrackLauncherImpl<F, T1, T2>
{
    F call_with_track;
    T1 arg1;
    T2 arg2;

    CELER_FUNCTION void operator()(CoreTrackView const& track) const
    {
        return this->call_with_track(track, arg1, arg2);
    }
};

//---------------------------------------------------------------------------//
template<class F, class T1, class T2, class T3>
struct TrackLauncherImpl<F, T1, T2, T3>
{
    F call_with_track;
    T1 arg1;
    T2 arg2;
    T3 arg3;

    CELER_FUNCTION void operator()(CoreTrackView const& track) const
    {
        return this->call_with_track(track, arg1, arg2, arg3);
    }
};

//---------------------------------------------------------------------------//
template<class F, class T1, class T2, class T3, class T4>
struct TrackLauncherImpl<F, T1, T2, T3, T4>
{
    F call_with_track;
    T1 arg1;
    T2 arg2;
    T3 arg3;
    T4 arg4;

    CELER_FUNCTION void operator()(CoreTrackView const& track) const
    {
        return this->call_with_track(track, arg1, arg2, arg3, arg4);
    }
};

//---------------------------------------------------------------------------//
// CONDITIONS
//---------------------------------------------------------------------------//
/*!
 * Condition for ConditionalTrackLauncher for active tracks.
 */
inline CELER_FUNCTION bool applies_alive(SimTrackView const& sim)
{
    return sim.status() == TrackStatus::alive;
}

//---------------------------------------------------------------------------//
/*!
 * Apply only to tracks with the given action ID.
 */
struct IsStepActionEqual
{
    ActionId action;

    CELER_FUNCTION bool operator()(SimTrackView const& sim) const
    {
        return sim.step_limit().action == this->action;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
