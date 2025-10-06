//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/detail/InitBoundaryExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/ArrayOperators.hh"
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
        return sim.post_step_action()
                   == track.surface_physics().scalars().init_boundary_action
               && sim.status() == TrackStatus::alive;
    }());

    auto geo = track.geometry();
    CELER_EXPECT(geo.is_on_boundary());

    // Surface selector must be created before crossing boundary to store
    // pre-volume information
    VolumeSurfaceSelector select_surface{track.surface(),
                                         geo.volume_instance_id()};
    OptMatId pre_volume_material = track.material_record().material_id();

    // Move the particle across the boundary
    geo.cross_boundary();
    if (CELER_UNLIKELY(geo.failed()))
    {
        track.apply_errored();
        return;
    }

    OptMatId post_volume_material = track.material_record().material_id();
    auto surface_physics = track.surface_physics();

    // Find oriented surface after crossing boundary using post-volume
    // information
    auto oriented_surface
        = select_surface(track.surface(), geo.volume_instance_id());
    if (!oriented_surface)
    {
        // Kill the track if the post-volume doesn't have a valid optical
        // material and there's no surface
        if (!post_volume_material)
        {
            track.sim().status(TrackStatus::killed);
            return;
        }

        // Use default surface data
        oriented_surface.surface = surface_physics.scalars().default_surface;
        oriented_surface.orientation = SubsurfaceDirection::forward;
    }

    // Enforce surface normal convention, swapping normal if geometry returns
    // one not entering the surface
    Real3 global_normal = geo.normal();
    if (!is_entering_surface(geo.dir(), global_normal))
    {
        global_normal = -global_normal;
    }

    surface_physics
        = SurfacePhysicsTrackView::Initializer{oriented_surface.surface,
                                               oriented_surface.orientation,
                                               global_normal,
                                               pre_volume_material,
                                               post_volume_material};

    CELER_ASSERT(
        is_entering_surface(geo.dir(), surface_physics.global_normal()));

    // TODO: replace with surface stepping action when implemented
    track.sim().post_step_action(
        surface_physics.scalars().post_boundary_action);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
