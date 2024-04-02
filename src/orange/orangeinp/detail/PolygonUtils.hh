//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/PolygonUtils.hh
//! \brief Utility standalone functions for polygons in 2D or 3D space.
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <vector>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
using Real2 = Array<real_type, 2>;
//---------------------------------------------------------------------------//
/*!
 *  Polygon orientation based on ordering of vertices.
 */
enum class Orientation
{
    clockwise = -1,
    collinear = 0,
    counterclockwise = 1,
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Find orientation of ordered vertices in 2D coordinates.
 */
CELER_FORCEINLINE_FUNCTION Orientation orientation(Real2 const& a,
                                                   Real2 const& b,
                                                   Real2 const& c)
{
    auto crossp = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (crossp == 0)
        return Orientation::collinear;  // collinear
    return (crossp < 0) ? Orientation::clockwise
                        : Orientation::counterclockwise;
}

//---------------------------------------------------------------------------//
/*! Check if a 2D polygon is convex.
 *
 * /param corners the vertices of the polygon
 * /param degen_ok if true, consecutive degenerate points are okay, still
 *   returns true
 */
/*explicit CELER_FUNCTION bool
is_convex(std::vector<const Real2> const& corners, bool degen_ok = false)
{
    is_convex(Span<const Real2>{static_cast<const Real2*>(corners.data()),
                                corners.size()},
              degen_ok);
}*/

CELER_FUNCTION bool
is_convex(Span<const Real2> const& corners, bool degen_ok = false)
{
    // The cross product of all vector pairs corresponding to ordered
    // consecutive segments has to be positive.
    auto crossp = [&](Real2 const& v1, Real2 const& v2) {
        return v1[0] * v2[1] - v1[1] * v2[0];
    };

    auto vecsub = [&](Real2 const& v1, Real2 const& v2) {
        return Real2{v1[0] - v2[0], v1[1] - v2[1]};
    };

    // Use the cross_product(last, first) as sign reference
    auto num_corners = corners.size();
    auto vec1 = vecsub(corners[0], corners[num_corners - 1]);
    auto vec2 = vecsub(corners[1], corners[0]);
    auto ref = crossp(vec1, vec2);
    for (auto i : range(num_corners - 1))
    {
        vec1 = vec2;
        vec2 = vecsub(corners[(i + 2) % num_corners],
                      corners[(i + 1) % num_corners]);
        auto val = crossp(vec1, vec2);
        // Make sure the sign is the same as the reference
        if (val * ref < 0.0 || (!degen_ok && val == 0.0))
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
/*! Check if a 3D polygon is planar
 *
 * /param a,b,c - the polygon vertices
 */
CELER_FUNCTION bool
is_planar(Real3 const& a, Real3 const& b, Real3 const& c, Real3 const& d)
{
    using celeritas::axpy;
    auto vecsub = [&](Real3 const& v1, Real3 const& v2) -> Real3 {
        Real3 result{v1};
        axpy(real_type{-1.0}, v2, &result);
        return result;
    };

    // use the cross_product(last, first) as sign reference
    auto norm = make_unit_vector(cross_product(vecsub(b, a), vecsub(c, a)));
    auto val = dot_product(norm, vecsub(d, a));
    return std::fabs(val) < Tolerance<real_type>::from_softequal();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
