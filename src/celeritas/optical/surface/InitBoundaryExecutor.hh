//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/InitBoundaryExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"
#include "celeritas/optical/Types.hh"

#include "SurfacePhysicsView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct InitBoundaryExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void InitBoundaryExecutor::operator()(CoreTrackView& track) const
{
    CELER_EXPECT([track] {
        auto sim = track.sim();
        return sim.post_step_action() == track.boundary_action()
               && sim.status() == TrackStatus::alive;
    }());

    auto geo = track.geometry();
    CELER_EXPECT(geo.is_on_boundary());

    auto pre_surface = track.volume_surface(geo.volume_id());
    auto pre_volume_inst = geo.volume_instance_id();

    // Move the particle across the boundary
    geo.cross_boundary();
    if (CELER_UNLIKELY(geo.failed()))
    {
        track.apply_errored();
        return;
    }
    else
    {
        auto post_volume = geo.volume_id();
        auto post_volume_inst = geo.volume_instance_id();

        // Lookup first by interface
        auto surface_id
            = pre_surface.find_interface(pre_volume_inst, post_volume_inst);
        if (!surface_id)
        {
            // Lookup pre-volume boundary
            surface_id = pre_surface.boundary_id();

            if (!surface_id)
            {
                // Lookup post-volume boundary
                surface_id = track.volume_surface(post_volume).boundary_id();
            }
        }

        if (!surface_id)
        {
            // If there's no surface, mark photon as killed
            track.sim().status(TrackStatus::killed);
        }
        else
        {
            // initialize surface state
            track.surface_physics()
                = SurfacePhysicsView::Initializer{surface_id, geo.normal()};
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
