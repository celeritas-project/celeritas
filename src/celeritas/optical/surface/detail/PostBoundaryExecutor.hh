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
    auto traverse = track.surface_physics().traversal();
    CELER_EXPECT(traverse.is_exiting());

    if (traverse.in_pre_volume())
    {
        // Re-entrant into the pre-volume
        auto geo = track.geometry();
        geo.cross_boundary();
        if (CELER_UNLIKELY(geo.failed()))
        {
            track.apply_errored();
            return;
        }
    }

    track.surface_physics().reset();

    if (!track.material_record().material_id())
    {
        // Kill track if it enters an invalid optical material
        track.sim().status(TrackStatus::killed);
    }

    CELER_ENSURE(!track.surface_physics().is_crossing_boundary());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
