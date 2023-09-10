//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SoftSurfaceEqual.cc
//---------------------------------------------------------------------------//
#include "SoftSurfaceEqual.hh"

#include <cmath>

#include "detail/AllSurfaces.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(S)                                         \
    template bool SoftSurfaceEqual::operator()(S<Axis::x> const&,        \
                                               S<Axis::x> const&) const; \
    template bool SoftSurfaceEqual::operator()(S<Axis::y> const&,        \
                                               S<Axis::y> const&) const; \
    template bool SoftSurfaceEqual::operator()(S<Axis::z> const&,        \
                                               S<Axis::z> const&) const

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Compare two aligned planes for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(PlaneAligned<T> const& a,
                                  PlaneAligned<T> const& b) const
{
    return soft_eq_(a.position(), b.position());
}

//! \cond
ORANGE_INSTANTIATE_OP(PlaneAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two centered axis-aligned cylinders for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(CylCentered<T> const& a,
                                  CylCentered<T> const& b) const
{
    return soft_eq_sq_(a.radius_sq(), b.radius_sq());
}

//! \cond
ORANGE_INSTANTIATE_OP(CylCentered);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two centered spheres for near equality.
 */
bool SoftSurfaceEqual::operator()(SphereCentered const& a,
                                  SphereCentered const& b) const
{
    return soft_eq_sq_(a.radius_sq(), b.radius_sq());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two aligned cylinders for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(CylAligned<T> const& a,
                                  CylAligned<T> const& b) const
{
    return soft_eq_sq_(a.radius_sq(), b.radius_sq())
           && soft_eq_distance_(a.calc_origin(), b.calc_origin());
}

//! \cond
ORANGE_INSTANTIATE_OP(CylAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two planes for near equality.
 */
bool SoftSurfaceEqual::operator()(Plane const& a, Plane const& b) const
{
    // Guard against dot product being slightly greater than 1 due to fp
    // arithmetic
    real_type const ndot = dot_product(a.normal(), b.normal());
    return (ndot >= 1 || std::sqrt(1 - ndot) < soft_eq_.rel() * 3)
           && soft_eq_(a.displacement(), b.displacement());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two spheres for near equality.
 */
bool SoftSurfaceEqual::operator()(Sphere const& a, Sphere const& b) const
{
    return soft_eq_sq_(a.radius_sq(), b.radius_sq())
           && soft_eq_distance_(a.origin(), b.origin());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two cones for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(ConeAligned<T> const& a,
                                  ConeAligned<T> const& b) const
{
    return soft_eq_sq_(a.tangent_sq(), b.tangent_sq())
           && soft_eq_distance_(a.origin(), b.origin());
}

//! \cond
ORANGE_INSTANTIATE_OP(ConeAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Compare two simple quadrics for near equality.
 */
bool SoftSurfaceEqual::operator()(SimpleQuadric const& a,
                                  SimpleQuadric const& b) const
{
    return soft_eq_distance_(make_array(a.second()), make_array(b.second()))
           && soft_eq_distance_(make_array(a.first()), make_array(b.first()))
           && soft_eq_(a.zeroth(), b.zeroth());
}

//---------------------------------------------------------------------------//
/*!
 * Compare two general quadrics for near equality.
 */
bool SoftSurfaceEqual::operator()(GeneralQuadric const& a,
                                  GeneralQuadric const& b) const
{
    return soft_eq_distance_(make_array(a.second()), make_array(b.second()))
           && soft_eq_distance_(make_array(a.cross()), make_array(b.cross()))
           && soft_eq_distance_(make_array(a.first()), make_array(b.first()))
           && soft_eq_(a.zeroth(), b.zeroth());
}

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Compare the square of values for soft equality.
 */
bool SoftSurfaceEqual::soft_eq_sq_(real_type a, real_type b) const
{
    return soft_eq_(std::sqrt(a), std::sqrt(b));
}

//---------------------------------------------------------------------------//
/*!
 * Compare the distance between two points for being less than the tolerance.
 */
bool SoftSurfaceEqual::soft_eq_distance_(Real3 const& a, Real3 const& b) const
{
    // This is soft equal formula but using vector distance.
    real_type rel = soft_eq_.rel() * std::fmax(norm(a), norm(b));
    return distance(a, b) < 3 * std::fmax(soft_eq_.abs(), rel);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
