//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "sim/CoreTrackData.hh"
#include "sim/CoreTrackView.hh"

#include "detail/TrackLauncherImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a function-like class to launch a "Track"-dependent function.
 *
 * This class should be used primarily by generated kernel functions:
 *
 * \code
__global__ void foo_kernel(
    CoreParamsDeviceRef const params,
    CoreStateDeviceRef const states)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    auto launch = make_track_launcher(params, states, foo_track);
    launch(tid);
}
\endcode
 *
 * where the user-written code defines an inline function using the core track
 * view:
 *
 * \code
inline CELER_FUNCTION void foo_track(celeritas::CoreTrackView const& track)
{
    // ...
}
   \endcode
 */
template<class F>
CELER_FUNCTION detail::TrackLauncher<F> make_track_launcher(
    CoreParamsData<Ownership::const_reference, MemSpace::native> const& params,
    CoreStateData<Ownership::reference, MemSpace::native> const&        state,
    F&& call_with_track)
{
    return {params, state, ::celeritas::forward<F>(call_with_track)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
