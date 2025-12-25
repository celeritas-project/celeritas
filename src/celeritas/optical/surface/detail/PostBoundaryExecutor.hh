//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/detail/PostBoundaryExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/optical/CoreTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#endif

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Finalize the track's boundary crossing.
 *
 * Updates the track's state base on whether it is re-entrant in the
 * pre-volume or entrant on the post-volume. The track's surface physics
 * state will be reset.
 */
struct PostBoundaryExecutor
{
    // Finalize track's boundary crossing
    inline CELER_FUNCTION void operator()(CoreTrackView&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Finalize the track's boundary crossing.
 */
CELER_FUNCTION void PostBoundaryExecutor::operator()(CoreTrackView& track) const
{
    auto geo = track.geometry();

    auto traverse = track.surface_physics().traversal();
    CELER_EXPECT(traverse.is_exiting());

    if (traverse.in_pre_volume())
    {
        // Reentrant into the pre-volume
        geo.cross_boundary();
        if (CELER_UNLIKELY(geo.failed()))
        {
            track.apply_errored();
            return;
        }
    }
    else
    {
        // Crossing into a new volume
        ImplVolumeId iv_id = geo.impl_volume_id();
        DetectorId det_id = track.detectors().detector_id(iv_id);
        if (det_id)
        {
            auto energy = track.particle().energy();
#if !CELER_DEVICE_COMPILE
            CELER_LOG_LOCAL(debug)
                << "hit volume " << iv_id.get() << " on detector "
                << det_id.get() << " with energy " << energy.value();
#endif
            track.sim().status(TrackStatus::killed);
        }
    }

    track.surface_physics().reset();

    CELER_ENSURE(!track.surface_physics().is_crossing_boundary());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
