//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.cc
//---------------------------------------------------------------------------//
#include "SimpleQuadric.hh"

#include "corecel/cont/Range.hh"

#include "ConeAligned.hh"
#include "CylAligned.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Promote from an axis-aligned cone.
 */
template<Axis T>
SimpleQuadric::SimpleQuadric(ConeAligned<T> const& other)
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

//---------------------------------------------------------------------------//
/*!
 * Promote from an axis-aligned cylinder.
 */
template<Axis T>
SimpleQuadric::SimpleQuadric(CylAligned<T> const& other)
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

//---------------------------------------------------------------------------//

template SimpleQuadric::SimpleQuadric(ConeAligned<Axis::x> const&);
template SimpleQuadric::SimpleQuadric(ConeAligned<Axis::y> const&);
template SimpleQuadric::SimpleQuadric(ConeAligned<Axis::z> const&);

template SimpleQuadric::SimpleQuadric(CylAligned<Axis::x> const&);
template SimpleQuadric::SimpleQuadric(CylAligned<Axis::y> const&);
template SimpleQuadric::SimpleQuadric(CylAligned<Axis::z> const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
