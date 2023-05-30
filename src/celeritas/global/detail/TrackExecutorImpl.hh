//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/TrackExecutorImpl.hh
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
// TrackExecutorImpl
//---------------------------------------------------------------------------//
//! Lazy implementation that replaces tuple + apply
template<class F, class... Ts>
struct TrackExecutorImpl;

//---------------------------------------------------------------------------//
template<class F>
struct TrackExecutorImpl<F>
{
    F call_with_track;

    CELER_FUNCTION void operator()(CoreTrackView const& track)
    {
        return this->call_with_track(track);
    }
};

//---------------------------------------------------------------------------//
template<class F, class T1>
struct TrackExecutorImpl<F, T1>
{
    F call_with_track;
    T1 arg1;

    CELER_FUNCTION void operator()(CoreTrackView const& track)
    {
        return this->call_with_track(track, celeritas::forward<T1>(arg1));
    }
};

//---------------------------------------------------------------------------//
template<class F, class T1, class T2>
struct TrackExecutorImpl<F, T1, T2>
{
    F call_with_track;
    T1 arg1;
    T2 arg2;

    CELER_FUNCTION void operator()(CoreTrackView const& track)
    {
        return this->call_with_track(
            track, celeritas::forward<T1>(arg1), celeritas::forward<T2>(arg2));
    }
};

//---------------------------------------------------------------------------//
template<class F, class T1, class T2, class T3>
struct TrackExecutorImpl<F, T1, T2, T3>
{
    F call_with_track;
    T1 arg1;
    T2 arg2;
    T3 arg3;

    CELER_FUNCTION void operator()(CoreTrackView const& track)
    {
        return this->call_with_track(track,
                                     celeritas::forward<T1>(arg1),
                                     celeritas::forward<T2>(arg2),
                                     celeritas::forward<T3>(arg3));
    }
};

//---------------------------------------------------------------------------//
template<class F, class T1, class T2, class T3, class T4>
struct TrackExecutorImpl<F, T1, T2, T3, T4>
{
    F call_with_track;
    T1 arg1;
    T2 arg2;
    T3 arg3;
    T4 arg4;

    CELER_FUNCTION void operator()(CoreTrackView const& track)
    {
        return this->call_with_track(track,
                                     celeritas::forward<T1>(arg1),
                                     celeritas::forward<T2>(arg2),
                                     celeritas::forward<T3>(arg3),
                                     celeritas::forward<T4>(arg4));
    }
};

//---------------------------------------------------------------------------//
// CONDITIONS
//---------------------------------------------------------------------------//
/*!
 * Condition for ConditionalTrackExecutor for active tracks.
 */
inline CELER_FUNCTION bool applies_active(SimTrackView const& sim)
{
    return sim.status() != TrackStatus::inactive;
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
/*!
 * Apply only to tracks with the given action ID.
 */
struct IsAlongStepActionEqual
{
    ActionId action;

    CELER_FUNCTION bool operator()(SimTrackView const& sim) const
    {
        CELER_EXPECT(applies_active(sim)
                     == static_cast<bool>(sim.along_step_action()));
        return sim.along_step_action() == this->action;
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
