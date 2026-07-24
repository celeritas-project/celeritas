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
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/surface/VolumeSurfaceSelector.hh"
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
 * Initialize a track for crossing a boundary that has surface physics enabled.
 *
 * The track is expected to be on a boundary in the pre-crossing volume, and is
 * then crosses the boundary to get the post-crossing volume. If a surface
 * exists between these volumes then the surface ID and normal are filled in
 * the track's surface state data. Otherwise the track is killed at the
 * surface.
 *
 * \note See and update documentation in \rstref{Boundary
 * initialization, surface_boundary_init} .
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
#if !CELER_DEVICE_COMPILE
    // Pre-crossing volume, for the boundary-selection tally below
    unsigned int const dbg_pre_vol = geo.volume_id().unchecked_get();
    unsigned int const dbg_pre_vi = geo.volume_instance_id().unchecked_get();
#endif

    // Move the particle across the boundary
    geo.cross_boundary();
    if (CELER_UNLIKELY(geo.failed()))
    {
        track.apply_errored();
        return;
    }
    if (CELER_UNLIKELY(geo.is_outside()))
    {
#if !CELER_DEVICE_COMPILE
        celeritas::optical::detail::tally_optical_kill(
            "escaped-world", 0,
            track.particle().energy().value() > 4.576e-6);
#endif
        track.sim().status(TrackStatus::killed);
        return;
    }
    OptMatId post_volume_material = track.material_record().material_id();
#if !CELER_DEVICE_COMPILE
    {
        // Debug: tally foil-related crossings (CCM deficit hunt)
        unsigned int post_vol = track.geometry().volume_id().unchecked_get();
        bool vis = track.particle().energy().value() <= 4.576e-6;
        if (vis)
        {
            if (post_vol == 6)
            {
                celeritas::optical::detail::tally_optical_kill(
                    "crossing-to-ptfe", post_vol, false);
            }
            else if (post_vol == 10)
                celeritas::optical::detail::tally_optical_kill(
                    "crossing-to-lar", post_vol, false);
            else if (post_vol >= 7 && post_vol <= 9)
                celeritas::optical::detail::tally_optical_kill(
                    "crossing-to-foil", post_vol, false);
        }
    }
#endif
    auto surface_physics = track.surface_physics();

    // Find oriented surface after crossing boundary using post-volume
    // information
    auto oriented_surface
        = select_surface(track.surface(), geo.volume_instance_id());
#if !CELER_DEVICE_COMPILE
    bool const dbg_defaulted = !oriented_surface;
#endif
    if (!oriented_surface)
    {
        // Use default surface properties: typically dielectric-dielectric
        oriented_surface.surface = surface_physics.scalars().default_surface;
        oriented_surface.orientation = LocalDirection::forward;
    }

#if !CELER_DEVICE_COMPILE
    if (track.particle().energy().value() <= 4.576e-6)
    {
        // Full boundary-selection tuple for visible photons: which surface
        // (and orientation) the selector picked for a given volume pair.
        // Diffing this between geometry drivers names any boundary whose
        // surface is resolved differently.
        char xbuf[96];
        std::snprintf(xbuf,
                      sizeof(xbuf),
                      "xing pre=%u vi=%u surf=%u%s%s",
                      dbg_pre_vol,
                      dbg_pre_vi,
                      oriented_surface.surface.unchecked_get(),
                      oriented_surface.orientation == LocalDirection::forward
                          ? "f"
                          : "r",
                      dbg_defaulted ? "D" : "");
        celeritas::optical::detail::tally_optical_kill(
            xbuf, track.geometry().volume_id().unchecked_get(), false);
    }
#endif

    // Enforce surface normal convention, swapping normal if geometry returns
    // one not entering the surface
    Real3 global_normal = geo.normal();
    if (!is_entering_surface(geo.dir(), global_normal))
    {
        global_normal = -global_normal;
    }

#if !CELER_DEVICE_COMPILE
    if (track.geometry().volume_id().unchecked_get() == 6
        && track.particle().energy().value() <= 4.576e-6)
    {
        char buf[64];
        std::snprintf(buf,
                      sizeof(buf),
                      "ptfe-surface-id%u-%s",
                      oriented_surface.surface.unchecked_get(),
                      oriented_surface.orientation == LocalDirection::forward
                          ? "fwd"
                          : "rev");
        celeritas::optical::detail::tally_optical_kill(buf, 6, false);
    }
#endif
    surface_physics = [&] {
        SurfacePhysicsTrackView::Initializer init;
        init.surface = oriented_surface.surface;
        init.orientation = oriented_surface.orientation;
        init.global_normal = global_normal;
        init.pre_volume_material = pre_volume_material;
        // Note that post-volume material may be null
        init.post_volume_material = post_volume_material;
        return init;
    }();

    CELER_ASSERT(
        is_entering_surface(geo.dir(), surface_physics.global_normal()));

#if !CELER_DEVICE_COMPILE
    if (celeritas::optical::detail::surface_trace_enabled())
    {
        auto& slot = celeritas::optical::detail::traced_slot();
        unsigned int post_vol = track.geometry().volume_id().unchecked_get();
        bool vis = track.particle().energy().value() <= 4.576e-6;
        int me = static_cast<int>(track.track_slot_id().get());
        if (vis && post_vol == 6)
        {
            int expect = -1;
            slot.compare_exchange_strong(expect, me);
        }
        if (me == slot.load())
        {
            char buf[192];
            std::snprintf(
                buf,
                sizeof(buf),
                "slot%d INIT post_vol=%u surf=%u %s dir_n=%.3f pos=%u nlp=%u "
                "xyz=(%.5f,%.5f,%.5f) dir=(%.3f,%.3f,%.3f)",
                me,
                post_vol,
                oriented_surface.surface.unchecked_get(),
                oriented_surface.orientation == LocalDirection::forward
                    ? "fwd"
                    : "rev",
                dot_product(geo.dir(), global_normal),
                surface_physics.traversal().pos().unchecked_get(),
                surface_physics.traversal().num_local_pos(),
                geo.pos()[0],
                geo.pos()[1],
                geo.pos()[2],
                geo.dir()[0],
                geo.dir()[1],
                geo.dir()[2]);
            celeritas::optical::detail::trace_surface(buf);
        }
    }
#endif

    track.sim().post_step_action(
        surface_physics.scalars().surface_stepping_action);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
