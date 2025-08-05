//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/detail/InitBoundaryExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/surface/VolumeSurfaceSelector.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize a track for crossing a boundary that has surface physics enabled.
 *
 * The track is expected to be on a boundary in the pre-crossing volume, and is
 * then crosses the boundary to get the post-crossing volume. If a surface
 * exists between these volumes then the surface ID and normal are filled in
 * the track's surface state data. Otherwise the track is killed at the
 * surface.
 */
struct InitBoundaryExecutor
{
    // Initialize track for boundary crossing
    inline CELER_FUNCTION void operator()(CoreTrackView& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize the track for boundary crossing.
 */
CELER_FUNCTION void InitBoundaryExecutor::operator()(CoreTrackView& track) const
{
    CELER_EXPECT([track] {
        auto sim = track.sim();
        return sim.post_step_action() == track.init_boundary_action()
               && sim.status() == TrackStatus::alive;
    }());

    auto geo = track.geometry();
    CELER_EXPECT(geo.is_on_boundary());

    // Surface selector must be created before crossing boundary to store
    // pre-volume information
    VolumeSurfaceSelector select_surface{track.surface(),
                                         geo.volume_instance_id()};

    // Move the particle across the boundary
    geo.cross_boundary();
    if (CELER_UNLIKELY(geo.failed()))
    {
        track.apply_errored();
        return;
    }

    // Find oriented surface after crossing boundary using post-volume
    // information
    if (auto oriented_surface
        = select_surface(track.surface(), geo.volume_instance_id()))
    {
        // initialize surface state
        track.sim().post_step_action(track.post_boundary_action());
    }
    else
    {
        // If there's no surface, mark photon as killed
        // TODO: Add default behavior
        track.sim().status(TrackStatus::killed);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
