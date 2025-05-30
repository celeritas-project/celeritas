//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SimpleReflectionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/ArrayOperators.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/optical/CoreTrackView.hh"

#include "SurfaceInteraction.hh"
#include "SurfacePhysicsData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct SimpleReflectionExecutor
{
    inline CELER_FUNCTION SurfaceInteraction operator()(CoreTrackView&) const;

    NativeCRef<SurfacePhysicsParamsData> params;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION SurfaceInteraction
SimpleReflectionExecutor::operator()(CoreTrackView& track) const
{
    // TODO: add assertions

    auto geo = track.geometry();
    auto const& direction = geo.dir();
    auto const& polarization = track.particle().polarization();
    auto rng = track.rng();

    // Get surface normal from geometry
    Real3 surface_normal = geo.global_surface_normal();

    real_type reflectivity = params.scalars.global_reflectivity;
    real_type refl_and_trans = reflectivity
                               + params.scalars.global_transmittance;

    CELER_EXPECT(0 <= reflectivity && reflectivity <= refl_and_trans
                 && refl_and_trans <= 1);

    real_type p = generate_canonical(rng);

    if (p < reflectivity)
    {
        // do simple reflection
        SurfaceInteraction result;

        result.action = SurfaceInteraction::Action::reflected;
        result.direction = direction
                           - 2 * dot_product(direction, surface_normal)
                                 * surface_normal;
        result.polarization = -polarization
                              + 2 * dot_product(polarization, surface_normal)
                                    * surface_normal;

        return result;
    }
    else if (p < refl_and_trans)
    {
        // do transmission
        SurfaceInteraction result;

        result.action = SurfaceInteraction::Action::transmitted;
        result.direction = direction;
        result.polarization = polarization;

        return result;
    }
    else
    {
        // do absorption
        return SurfaceInteraction::from_absorption();
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
