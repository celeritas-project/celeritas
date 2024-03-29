//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/LengthUnits.hh
//! \brief NOTE: only use inside geocel; prefer celeritas/Units.hh
//---------------------------------------------------------------------------//
#pragma once

//#define CELER_ICRT inline constexpr real_type
#include <iostream>
#include <vector>
#include <corecel/cont/Array.hh>
#include <corecel/math/ArrayUtils.hh>
#include <orange/OrangeTypes.hh>

namespace celeritas
{
namespace geoutils
{

template<typename T>
using Real2 = Array<T, 2>;
template<typename T>
using VecReal2 = std::vector<Real2<T>>;
template<typename T>
using Real3 = Array<T, 3>;
template<typename T>
using VecReal3 = std::vector<Real3<T>>;

enum PgonOrient : int
{
    clockwise = -1,
    collinear = 0,
    counterclockwise = 1,
    kCW = clockwise,
    kLine = collinear,
    kCCW = counterclockwise
};

//---------------------------------------------------------------------------//
// Find orientation of ordered triplet (a, b, c).
// The function returns following values
//  0 --> input points are collinear
// -1 --> Clockwise
// +1 --> Counterclockwise
template<typename T>
CELER_FUNCTION PgonOrient orientation_2D(Real2<T> const& a,
                                         Real2<T> const& b,
                                         Real2<T> const& c)
{
    T crossp = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (crossp == 0)
        return collinear;  // collinear
    return (crossp < 0) ? kCW : kCCW;  // clockwise or counterclockwise
}

//---------------------------------------------------------------------------//
/*! Check if a 2D polygon is convex.
 *
 * /param corners the vertices of the polygon
 * /param degen_ok if true, consecutive degenerate points are okay, still
 *   returns true
 */
template<typename T>
CELER_FUNCTION bool
is_convex_2D(VecReal2<T> const& corners, bool degen_ok = false)
{
    // The cross product of all vector pairs corresponding to ordered
    // consecutive segments has to be positive.
    auto crossp = [&](Real2<T> const& v1, Real2<T> const& v2) {
        return v1[0] * v2[1] - v1[1] * v2[0];
    };

    auto vecsub = [&](Real2<T> const& v1, Real2<T> const& v2) {
        return Real2<T>{v1[0] - v2[0], v1[1] - v2[1]};
    };

    // use the cross_product(last, first) as sign reference
    auto N = corners.size();
    auto vec1 = vecsub(corners[0], corners[N - 1]);
    auto vec2 = vecsub(corners[1], corners[0]);
    auto ref = crossp(vec1, vec2);
    for (size_type i = 0; i < N - 1; ++i)
    {
        vec1 = vec2;
        vec2 = vecsub(corners[(i + 2) % N], corners[(i + 1) % N]);
        T val = crossp(vec1, vec2);
        // make sure the sign is the same as the reference
        if (val * ref < T{0} || (!degen_ok && val == T{0}))
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
/*! Check if a 3D polygon is planar
 *
 * /param a,b,c - corners the vertices of the polygon
 */
template<typename T>
CELER_FUNCTION bool is_planar_3D(Real3<T> const& a,
                                 Real3<T> const& b,
                                 Real3<T> const& c,
                                 Real3<T> const& d)
{
    using celeritas::axpy;
    auto vecsub = [&](Real3<T> const& v1, Real3<T> const& v2) -> Real3<T> {
        Real3<T> result{v1};
        axpy(-1.0, v2, &result);
        return result;
    };

    // use the cross_product(last, first) as sign reference
    auto norm = make_unit_vector(cross_product(vecsub(b, a), vecsub(c, a)));
    auto val = dot_product(norm, vecsub(d, a));
    // return std::fabs(val) < Tolerance<T>::from_default(); // undef symbol
    return std::fabs(val) < 1e-8;
}

//---------------------------------------------------------------------------//
}  // namespace geoutils
}  // namespace celeritas
