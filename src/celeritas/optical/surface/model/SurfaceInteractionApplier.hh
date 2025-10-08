//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SurfaceInteractionApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/sys/KernelTraits.hh"
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
 * The functor \c F must take a \c CoreTrackview and return a \c
 * SurfaceInteraction.
 */
template<class F>
struct SurfaceInteractionApplierBaseImpl
{
    F sample_interaction;

    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
/*!
 * This class is partially specialized with a second template argument to
 * extract any launch bounds from the functor class.
 *
 * \todo Generalize this with the core interaction applier
 */
template<class F, typename = void>
struct SurfaceInteractionApplier : public SurfaceInteractionApplierBaseImpl<F>
{
    CELER_FUNCTION SurfaceInteractionApplier(F&& f)
        : SurfaceInteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

template<class F>
struct SurfaceInteractionApplier<F,
                                 std::enable_if_t<kernel_max_blocks_min_warps<F>>>
    : public SurfaceInteractionApplierBaseImpl<F>
{
    static constexpr int max_block_size = F::max_block_size;
    static constexpr int min_warps_per_eu = F::min_warps_per_eu;

    CELER_FUNCTION SurfaceInteractionApplier(F&& f)
        : SurfaceInteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
};

template<class F>
struct SurfaceInteractionApplier<F, std::enable_if_t<kernel_max_blocks<F>>>
    : public SurfaceInteractionApplierBaseImpl<F>
{
    static constexpr int max_block_size = F::max_block_size;

    CELER_FUNCTION SurfaceInteractionApplier(F&& f)
        : SurfaceInteractionApplierBaseImpl<F>{celeritas::forward<F>(f)}
    {
    }
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
SurfaceInteractionApplierBaseImpl<F>::operator()(CoreTrackView const& track) const
{
    // Sample interaction
    SurfaceInteraction result = this->sample_interaction(track);

    if (result.action == SurfaceInteraction::Action::absorbed)
    {
        // Mark particle as killed
        track.sim().status(TrackStatus::killed);
    }
    else
    {
        // Cross boundary if refracted
        auto surface_physics = track.surface_physics();
        auto traverse = surface_physics.traversal();
        if (result.action == SurfaceInteraction::Action::refracted)
        {
            traverse.cross_interface(traverse.dir());
        }

        // Update direction and polarization
        track.geometry().set_dir(result.direction);
        track.particle().polarization(result.polarization);
        surface_physics.update_traversal_direction(result.direction);

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
