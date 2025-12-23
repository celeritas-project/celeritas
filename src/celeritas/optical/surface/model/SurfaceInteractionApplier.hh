//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SurfaceInteractionApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"

#include "SurfaceInteraction.hh"

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
CELER_FUNCTION void
SurfaceInteractionApplier<F>::operator()(CoreTrackView const& track) const
{
    // Sample interaction
    SurfaceInteraction result = this->sample_interaction(track);

    CELER_ASSERT(result.is_valid());

    if (result.action == SurfaceInteraction::Action::absorbed)
    {
        // Mark particle as killed
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
            // Update direction and polarization
            track.geometry().set_dir(result.direction);
            track.particle().polarization(result.polarization);
            surface_physics.update_traversal_direction(result.direction);
        }

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
