//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.cc
//---------------------------------------------------------------------------//
#include "SimpleQuadric.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"

#include "ConeAligned.hh"
#include "CylAligned.hh"
#include "Plane.hh"
#include "Sphere.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with coefficients.
 */
SimpleQuadric::SimpleQuadric(Real3 const& abc, Real3 const& def, real_type g)
    : a_(abc[0])
    , b_(abc[1])
    , c_(abc[2])
    , d_(def[0])
    , e_(def[1])
    , f_(def[2])
    , g_(g)
{
    CELER_EXPECT(a_ != 0 || b_ != 0 || c_ != 0 || d_ != 0 || e_ != 0
                 || f_ != 0);
}

//---------------------------------------------------------------------------//
/*!
 * Promote from a plane.
 *
 * Note that the plane is written as \f$ ax + by + cz - d = 0 \f$
 * whereas the simple quadric has a different sign for the constant:
 * \f$ dx + ey + fz + g = 0 \f$ .
 */
SimpleQuadric::SimpleQuadric(Plane const& other) noexcept(!CELERITAS_DEBUG)
    : SimpleQuadric{{0, 0, 0}, other.normal(), negate(other.displacement())}
{
}

//---------------------------------------------------------------------------//
/*!
 * Promote from an axis-aligned cylinder.
 */
template<Axis T>
SimpleQuadric::SimpleQuadric(CylAligned<T> const& other) noexcept
{
    constexpr auto U = CylAligned<T>::u_axis();
    constexpr auto V = CylAligned<T>::v_axis();

    Real3 second{0, 0, 0};
    second[to_int(U)] = 1;
    second[to_int(V)] = 1;

    Real3 first{0, 0, 0};
    first[to_int(U)] = -2 * other.origin_u();
    first[to_int(V)] = -2 * other.origin_v();

    real_type zeroth = -other.radius_sq();
    zeroth += ipow<2>(other.origin_u());
    zeroth += ipow<2>(other.origin_v());

    *this = SimpleQuadric{second, first, zeroth};
}

//! \cond
template SimpleQuadric::SimpleQuadric(CylAligned<Axis::x> const&) noexcept;
template SimpleQuadric::SimpleQuadric(CylAligned<Axis::y> const&) noexcept;
template SimpleQuadric::SimpleQuadric(CylAligned<Axis::z> const&) noexcept;
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Promote from a sphere.
 *
 * \verbatim
   x^2 + y^2 + z^2
       - 2x_0 - 2y_0 - 2z_0
       + x_0^2 + y_0^2 + z_0^2 - r^2 = 0
 * \endverbatim
 */
SimpleQuadric::SimpleQuadric(Sphere const& other) noexcept(!CELERITAS_DEBUG)
{
    Real3 const& origin = other.origin();

    real_type zeroth = -other.radius_sq();
    for (auto ax : range(to_int(Axis::size_)))
    {
        zeroth += ipow<2>(origin[ax]);
    }

    *this = SimpleQuadric{{1, 1, 1}, real_type{-2} * origin, zeroth};
}

//---------------------------------------------------------------------------//
/*!
 * Promote from an axis-aligned cone.
 */
template<Axis T>
SimpleQuadric::SimpleQuadric(ConeAligned<T> const& other) noexcept
{
    constexpr auto U = ConeAligned<T>::u_axis();
    constexpr auto V = ConeAligned<T>::v_axis();
    Real3 const& origin = other.origin();

    Real3 second;
    second[to_int(T)] = -other.tangent_sq();
    second[to_int(U)] = 1;
    second[to_int(V)] = 1;

    Real3 first;
    first[to_int(T)] = 2 * origin[to_int(T)] * other.tangent_sq();
    first[to_int(U)] = -2 * origin[to_int(U)];
    first[to_int(V)] = -2 * origin[to_int(V)];

    real_type zeroth = -other.tangent_sq() * ipow<2>(origin[to_int(T)]);
    zeroth += ipow<2>(origin[to_int(U)]);
    zeroth += ipow<2>(origin[to_int(V)]);

    *this = SimpleQuadric{second, first, zeroth};
}

//! \cond
template SimpleQuadric::SimpleQuadric(ConeAligned<Axis::x> const&) noexcept;
template SimpleQuadric::SimpleQuadric(ConeAligned<Axis::y> const&) noexcept;
template SimpleQuadric::SimpleQuadric(ConeAligned<Axis::z> const&) noexcept;
//! \endcond

//---------------------------------------------------------------------------//
}  // namespace celeritas
