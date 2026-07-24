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
#include <cstdio>
#include "celeritas/optical/detail/OpticalKillTally.hh"

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
 *
 * \note This is only called if the traversal state is "exiting", as set by
 * SurfaceInteractionApplier .
 *
 * \sa BoundaryAction
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

#if !CELER_DEVICE_COMPILE
    if (celeritas::optical::detail::surface_trace_enabled()
        && static_cast<int>(track.track_slot_id().get())
               == celeritas::optical::detail::traced_slot().load())
    {
        char buf[160];
        std::snprintf(buf,
                      sizeof(buf),
                      "slot%d POST pos=%u in_pre=%d vol=%u",
                      static_cast<int>(track.track_slot_id().get()),
                      traverse.pos().unchecked_get(),
                      static_cast<int>(traverse.in_pre_volume()),
                      track.geometry().volume_id().unchecked_get());
        celeritas::optical::detail::trace_surface(buf);
    }
#endif

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
        // Kill track if it enters an invalid optical material after crossing
        // through a custom physics surface
#if !CELER_DEVICE_COMPILE
        celeritas::optical::detail::tally_optical_kill(
            "nonoptical-material",
            track.geometry().volume_id().unchecked_get(),
            track.particle().energy().value() > 4.576e-6);
#endif
        track.sim().status(TrackStatus::killed);
    }

    CELER_ENSURE(!track.surface_physics().is_crossing_boundary());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
