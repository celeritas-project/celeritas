//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/OnlyReflectionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/optical/CoreTrackView.hh"

#include "OnlyReflectionData.hh"
#include "ReflectionFormCalculator.hh"
#include "SurfaceInteraction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Return only a reflection interaction based on the \c ReflectionMode.
 *
 * This applies the \c ReflectionFormCalculator directly to the track for the
 * given mode.
 */
struct OnlyReflectionExecutor
{
    NativeCRef<OnlyReflectionData> data;

    inline CELER_FUNCTION SurfaceInteraction
    operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate surface interaction based on the reflection mode.
 */
CELER_FUNCTION SurfaceInteraction
OnlyReflectionExecutor::operator()(CoreTrackView const& track) const
{
    auto sub_model_id = track.surface_physics()
                            .interface(SurfacePhysicsOrder::interaction)
                            .internal_surface_id();
    CELER_ASSERT(sub_model_id < data.modes.size());

    ReflectionFormCalculator calc_reflection{
        track.geometry().dir(), track.particle(), track.surface_physics()};

    switch (data.modes[sub_model_id])
    {
        case ReflectionMode::specular_spike:
            return calc_reflection.calc_specular_spike();
        case ReflectionMode::specular_lobe:
            return calc_reflection.calc_specular_lobe();
        case ReflectionMode::backscatter:
            return calc_reflection.calc_backscatter();
        case ReflectionMode::diffuse_lobe: {
            auto rng = track.rng();
            return calc_reflection.sample_lambertian_reflection(rng);
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
