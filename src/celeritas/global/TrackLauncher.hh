//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/TrackLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "CoreTrackData.hh"

#include "celeritas/track/detail/TrackLauncherImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a function-like class to launch a "Track"-dependent function.
 *
 * This class should be used primarily by generated kernel functions:
 *
 * \code
__global__ void foo_kernel(CoreDeviceRef const data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    auto launch = make_track_launcher(data, foo_track);
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
CELER_FUNCTION detail::TrackLauncher<F>
make_track_launcher(CoreRef<MemSpace::native> const& data, F&& call_with_track)
{
    return {data, ::celeritas::forward<F>(call_with_track)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
