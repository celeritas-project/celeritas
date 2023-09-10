//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SoftSurfaceEqual.cc
//---------------------------------------------------------------------------//
#include "SoftSurfaceEqual.hh"

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
    return &a == &b;
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
    return &a == &b;
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
    return &a == &b;
}

//---------------------------------------------------------------------------//
/*!
 * Compare two aligned cylinders for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(CylAligned<T> const& a,
                                  CylAligned<T> const& b) const
{
    return &a == &b;
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
    return &a == &b;
}

//---------------------------------------------------------------------------//
/*!
 * Compare two spheres for near equality.
 */
bool SoftSurfaceEqual::operator()(Sphere const& a, Sphere const& b) const
{
    return &a == &b;
}

//---------------------------------------------------------------------------//
/*!
 * Compare two cones for near equality.
 */
template<Axis T>
bool SoftSurfaceEqual::operator()(ConeAligned<T> const& a,
                                  ConeAligned<T> const& b) const
{
    return &a == &b;
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
    return &a == &b;
}

//---------------------------------------------------------------------------//
/*!
 * Compare two general quadrics for near equality.
 */
bool SoftSurfaceEqual::operator()(GeneralQuadric const& a,
                                  GeneralQuadric const& b) const
{
    return &a == &b;
}

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Compare the distance between two points for being less than the tolerance.
 */
bool SoftSurfaceEqual::soft_eq_distance_(Real3 const& a, Real3 const& b) const
{
    // This is soft equal formula but using vector distance.
    real_type rel = soft_eq_.rel() * std::fmax(norm(a), norm(b));
    return distance(a, b) < std::fmax(soft_eq_.abs(), rel);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
