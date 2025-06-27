//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceInteractionApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/sys/KernelTraits.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"

#include "SurfaceInteraction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a surface interaction executor and apply it to a track.
 *
 * The function F must take a \c CoreTrackView and return a \c
 * SurfaceInteraction
 */
template<class F>
struct SurfaceInteractionApplierBaseImpl
{
    F sample_interaction;

    CELER_FUNCTION void operator()(CoreTrackView&);
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
 * Sample a surface interaction and apply to the track view.
 *
 * The given track *must* be an active track with the correct step limit action
 * ID.
 */
template<class F>
CELER_FUNCTION void
SurfaceInteractionApplierBaseImpl<F>::operator()(CoreTrackView& track)
{
    SurfaceInteraction result = this->sample_interaction(track);

    if (result.action == SurfaceInteraction::Action::absorbed)
    {
        // Mark particle as killed
        track.sim().status(TrackStatus::killed);
    }
    else
    {
        // Update direction and polarization
        track.geometry().set_dir(result.direction);
        track.particle().polarization(result.polarization);

        if (result.action == SurfaceInteraction::Action::transmitted)
        {
            auto surface = track.surface();

            if (dot_product(track.geometry().dir(), surface.surface_normal())
                < 0)
            {
                // Decrement layer

                if (surface.current_layer() == SurfaceLayerId{0})
                {
                    // Exit boundary process
                }
                else
                {
                    surface.current_layer()--;
                }
            }
            else
            {
                // Increment layer

                if (surface.current_layer()
                    == SurfaceLayerId{surface.num_layers() - 1})
                {
                    // Exit boundary process
                }
                else
                {
                    surface.current_layer()++;
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
