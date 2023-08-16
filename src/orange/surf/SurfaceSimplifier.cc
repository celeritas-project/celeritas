//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceSimplifier.cc
//---------------------------------------------------------------------------//
#include "SurfaceSimplifier.hh"

#include "corecel/math/SoftEqual.hh"

#include "ConeAligned.hh"
#include "CylAligned.hh"
#include "CylCentered.hh"
#include "GeneralQuadric.hh"
#include "Plane.hh"
#include "PlaneAligned.hh"
#include "SimpleQuadric.hh"
#include "Sphere.hh"
#include "SphereCentered.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(OUT, IN)                       \
    template SurfaceSimplifier::Optional<OUT<Axis::x>>       \
    SurfaceSimplifier::operator()(IN<Axis::x> const&) const; \
    template SurfaceSimplifier::Optional<OUT<Axis::y>>       \
    SurfaceSimplifier::operator()(IN<Axis::y> const&) const; \
    template SurfaceSimplifier::Optional<OUT<Axis::z>>       \
    SurfaceSimplifier::operator()(IN<Axis::z> const&) const

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Plane may be snapped to origin.
 */
template<Axis T>
auto SurfaceSimplifier::operator()(PlaneAligned<T> const& p) const
    -> Optional<PlaneAligned<T>>
{
    if (p.position() != real_type{0} && SoftZero{tol_}(p.position()))
    {
        // Snap to zero since it's not already zero
        return PlaneAligned<T>{real_type{0}};
    }
    // No simplification performed
    return {};
}

ORANGE_INSTANTIATE_OP(PlaneAligned, PlaneAligned);

//---------------------------------------------------------------------------//
/*!
 * Cylinder at origin will be simplified.
 */
template<Axis T>
auto SurfaceSimplifier::operator()(CylAligned<T> const&) const
    -> Optional<CylCentered<T>>
{
    return {};
}

ORANGE_INSTANTIATE_OP(CylCentered, CylAligned);

//---------------------------------------------------------------------------//
/*!
 * Plane may be flipped, adjusted, or become axis-aligned.
 */
auto SurfaceSimplifier::operator()(Plane const&) -> Optional<Plane>
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Sphere near center can be snapped.
 */
auto SurfaceSimplifier::operator()(Sphere const&) const
    -> Optional<SphereCentered>
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Simple quadric with near-zero terms can be another second-order surface.
 */
auto SurfaceSimplifier::operator()(SimpleQuadric const&)
    -> Optional<Sphere,
                ConeAligned<Axis::x>,
                ConeAligned<Axis::y>,
                ConeAligned<Axis::z>,
                CylAligned<Axis::x>,
                CylAligned<Axis::y>,
                CylAligned<Axis::z>>
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Quadric with no cross terms is simple.
 */
auto SurfaceSimplifier::operator()(GeneralQuadric const&)
    -> Optional<SimpleQuadric>
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
