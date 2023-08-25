//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/SurfaceTranslator.cc
//---------------------------------------------------------------------------//
#include "SurfaceTranslator.hh"

#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/MatrixUtils.hh"

#include "../ConeAligned.hh"
#include "../CylAligned.hh"
#include "../CylCentered.hh"
#include "../GeneralQuadric.hh"
#include "../Plane.hh"
#include "../PlaneAligned.hh"
#include "../SimpleQuadric.hh"
#include "../Sphere.hh"
#include "../SphereCentered.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
#define ORANGE_INSTANTIATE_OP(OUT, IN)                                      \
    template OUT<Axis::x> SurfaceTranslator::operator()(IN<Axis::x> const&) \
        const;                                                              \
    template OUT<Axis::y> SurfaceTranslator::operator()(IN<Axis::y> const&) \
        const;                                                              \
    template OUT<Axis::z> SurfaceTranslator::operator()(IN<Axis::z> const&) \
        const

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct a translated axis-aligned plane.
 */
template<Axis T>
PlaneAligned<T>
SurfaceTranslator::operator()(PlaneAligned<T> const& other) const
{
    real_type origin = tr_.translation()[to_int(T)];
    return PlaneAligned<T>{other.position() + origin};
}

//! \cond
ORANGE_INSTANTIATE_OP(PlaneAligned, PlaneAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a translated axis-aligned cylinder.
 */
template<Axis T>
CylAligned<T> SurfaceTranslator::operator()(CylCentered<T> const& other) const
{
    return (*this)(CylAligned<T>{other});
}

//! \cond
ORANGE_INSTANTIATE_OP(CylAligned, CylCentered);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a translated sphere.
 */
Sphere SurfaceTranslator::operator()(SphereCentered const& other) const
{
    return (*this)(Sphere{other});
}

//---------------------------------------------------------------------------//
/*!
 * Construct a translated axis-aligned cylinder.
 */
template<Axis T>
CylAligned<T> SurfaceTranslator::operator()(CylAligned<T> const& other) const
{
    return CylAligned<T>::from_radius_sq(tr_.transform_up(other.calc_origin()),
                                         other.radius_sq());
}

//! \cond
ORANGE_INSTANTIATE_OP(CylAligned, CylAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a translated general plane.
 */
Plane SurfaceTranslator::operator()(Plane const& other) const
{
    return Plane{
        other.normal(),
        other.displacement() + dot_product(tr_.translation(), other.normal())};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a translated sphere.
 */
Sphere SurfaceTranslator::operator()(Sphere const& other) const
{
    return Sphere::from_radius_sq(tr_.transform_up(other.origin()),
                                  other.radius_sq());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a translated cone.
 */
template<Axis T>
ConeAligned<T> SurfaceTranslator::operator()(ConeAligned<T> const& other) const
{
    return ConeAligned<T>::from_tangent_sq(tr_.transform_up(other.origin()),
                                           other.tangent_sq());
}

//! \cond
ORANGE_INSTANTIATE_OP(ConeAligned, ConeAligned);
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Construct a translated simple quadric.
 */
SimpleQuadric SurfaceTranslator::operator()(SimpleQuadric const& other) const
{
    auto const second = make_array(other.second());
    auto first = make_array(other.first());
    real_type zeroth = other.zeroth();
    auto const& origin = tr_.translation();

    // Expand out origin into the other terms
    for (auto i = to_int(Axis::x); i < to_int(Axis::size_); ++i)
    {
        first[i] -= 2 * second[i] * origin[i];
        zeroth += second[i] * ipow<2>(origin[i])
                  - 2 * other.first()[i] * origin[i];
    }
    return SimpleQuadric{second, first, zeroth};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a translated quadric.
 *
 * See celeritas-doc/nb/geo/quadric-transform.ipynb . The implementation below
 * is less than optimal because we don't need to explicitly construct the Q
 * matrix.
 */
GeneralQuadric SurfaceTranslator::operator()(GeneralQuadric const& other) const
{
    constexpr auto X = to_int(Axis::x);
    constexpr auto Y = to_int(Axis::y);
    constexpr auto Z = to_int(Axis::z);

    Real3 const second = make_array(other.second());
    Real3 const cross = make_array(other.cross()) / real_type(2);
    Real3 const first = make_array(other.first()) / real_type(2);

    // Nonlinear components of the quadric matrix
    SquareMatrix<real_type, 3> nonl{Real3{second[X], cross[X], cross[Z]},
                                    Real3{cross[X], second[Y], cross[Y]},
                                    Real3{cross[Z], cross[Y], second[Z]}};

    // Calculate q' = - Q t + q
    Real3 newfirst
        = gemv(real_type(-1), nonl, tr_.translation(), real_type(1), first);

    // Update constant:
    // j' = j - t*(q' + q)
    real_type newzeroth = other.zeroth()
                          - dot_product(tr_.translation(), newfirst + first);

    return GeneralQuadric{
        second, make_array(other.cross()), real_type(2) * newfirst, newzeroth};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
