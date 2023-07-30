//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.cc
//---------------------------------------------------------------------------//
#include "SimpleQuadric.hh"

#include "corecel/cont/Range.hh"

#include "CylAligned.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Promote from an axis-aligned cylinder.
 */
template<Axis T>
SimpleQuadric::SimpleQuadric(CylAligned<T> const& other)
{
    Real3 second{1, 1, 1};
    Real3 first{0, 0, 0};
    real_type zeroth = -other.radius_sq();

    second[to_int(T)] = 0;
    first[to_int(other.u_axis())] -= 2 * other.origin_u();
    first[to_int(other.v_axis())] -= 2 * other.origin_v();
    zeroth += ipow<2>(other.origin_u());
    zeroth += ipow<2>(other.origin_v());

    *this = SimpleQuadric{second, first, zeroth};
}

//---------------------------------------------------------------------------//

template SimpleQuadric::SimpleQuadric(CylAligned<Axis::x> const&);
template SimpleQuadric::SimpleQuadric(CylAligned<Axis::y> const&);
template SimpleQuadric::SimpleQuadric(CylAligned<Axis::z> const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
