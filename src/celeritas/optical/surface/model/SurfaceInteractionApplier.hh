//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SurfaceInteractionApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"

#include "SurfaceInteraction.hh"
#include "celeritas/optical/detail/OpticalKillTally.hh"
#include "corecel/math/ArrayUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a surface interaction executor and apply it to a track.
 *
 * The functor \c F must take a \c CoreTrackView and return a \c
 * SurfaceInteraction.
 */
template<class F>
struct SurfaceInteractionApplier
{
    F sample_interaction;

    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class F>
CELER_FUNCTION SurfaceInteractionApplier(F&&) -> SurfaceInteractionApplier<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply sampled surface interaction to the track.
 */
template<class F>
CELER_FUNCTION void SurfaceInteractionApplier<F>::operator()(
    CoreTrackView const& track) const
{
    // Sample interaction
    SurfaceInteraction result = this->sample_interaction(track);

    CELER_ASSERT(result.is_valid());

#if !CELER_DEVICE_COMPILE
    if (celeritas::optical::detail::surface_trace_enabled()
        && static_cast<int>(track.track_slot_id().get())
               == celeritas::optical::detail::traced_slot().load())
    {
        auto s_phys = track.surface_physics();
        char buf[192];
        std::snprintf(buf,
                      sizeof(buf),
                      "slot%d INTERACT action=%d pos=%u inc_n=%.3f out_n=%.3f",
                      static_cast<int>(track.track_slot_id().get()),
                      static_cast<int>(result.action),
                      s_phys.traversal().pos().unchecked_get(),
                      dot_product(track.geometry().dir(),
                                  s_phys.global_normal()),
                      dot_product(result.direction, s_phys.global_normal()));
        celeritas::optical::detail::trace_surface(buf);
    }
#endif

    if (result.action == SurfaceInteraction::Action::absorbed)
    {
        // Mark particle as killed
#if !CELER_DEVICE_COMPILE
        celeritas::optical::detail::tally_optical_kill(
            "surface-absorbed",
            track.geometry().volume_id().unchecked_get(),
            track.particle().energy().value() > 4.576e-6);
#endif
        track.sim().status(TrackStatus::killed);
    }
    else
    {
        // Cross boundary if refracted or transmitted
        auto surface_physics = track.surface_physics();
        auto traverse = surface_physics.traversal();
        if (result.action != SurfaceInteraction::Action::reflected)
        {
            traverse.cross_interface(traverse.dir());
        }

        if (result.action != SurfaceInteraction::Action::transmitted)
        {
#if !CELER_DEVICE_COMPILE
            if (result.action == SurfaceInteraction::Action::reflected)
            {
                double d = dot_product(result.direction,
                                       track.surface_physics().global_normal());
                celeritas::optical::detail::tally_optical_kill(
                    d < 0 ? "reflect-back" : "reflect-into-surface",
                    track.geometry().volume_id().unchecked_get(),
                    track.particle().energy().value() > 4.576e-6);
            }
#endif
            // Update direction and polarization
            track.geometry().set_dir(result.direction);
            track.particle().polarization(result.polarization);
            surface_physics.update_traversal_direction(result.direction);
        }

        // Ensure no other interactions are taken this step
        // TODO: switch to more general surface crossing status?
        surface_physics.reflectivity_action(ReflectivityAction::transmit);

        if (traverse.is_exiting())
        {
            // End boundary crossing if exiting
            track.sim().post_step_action(
                surface_physics.scalars().post_boundary_action);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
