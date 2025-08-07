//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Whether a track is entering the surface defined by the given normal.
 *
 * The surface normal convention used in Celeritas optical physics is that
 * the normal direction points opposite the incident track direction. This
 * function makes checks for this condition explicit in the code.
 */
inline CELER_FUNCTION bool
is_entering_surface(Real3 const& normal, Real3 const& dir)
{
    return dot_product(normal, dir) < 0;
}

//---------------------------------------------------------------------------//
/*!
 * Sample facet normal until the track direction is entering the surface.
 *
 * Some facet normal calculators might not produce surface normals valid for
 * optical physics surface crossings (see \c is_entering_surface ). This
 * functor will repeatedly sample the distribution until a valid facet normal
 * is sampled.
 */
template<class Calculator>
struct EnteringSurfaceNormalSampler
{
    Real3 const& dir;
    Calculator sample_normal;

    // Repeatedly sample facet normal until satisfies entering surface
    template<class Engine>
    CELER_FUNCTION Real3 operator()(Engine& rng)
    {
        Real3 local_normal;
        do
        {
            local_normal = sample_normal(rng);
        } while (!is_entering_surface(local_normal, dir));
        return local_normal;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
