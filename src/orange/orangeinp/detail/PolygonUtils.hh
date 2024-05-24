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
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
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
inline Orientation
calc_orientation(Real2 const& a, Real2 const& b, Real2 const& c)
{
    auto crossp = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    return crossp < 0   ? Orientation::clockwise
           : crossp > 0 ? Orientation::counterclockwise
                        : Orientation::collinear;
}

//---------------------------------------------------------------------------//
/*!
 * Test whether a 2D polygon has the given orientation.
 *
 * The list of input corners must have at least 3 points to be a polygon.
 */
inline bool has_orientation(Span<Real2 const> const& corners, Orientation o)
{
    CELER_EXPECT(corners.size() > 2);
    for (auto i : range(corners.size()))
    {
        auto j = (i + 1) % corners.size();
        auto k = (i + 2) % corners.size();
        if (calc_orientation(corners[i], corners[j], corners[k]) != o)
            return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Check if a 2D polygon is convex.
 *
 * \param corners the vertices of the polygon
 * \param degen_ok allow consecutive degenerate points
 */
bool is_convex(Span<Real2 const> const& corners, bool degen_ok = false)
{
    CELER_EXPECT(!corners.empty());

    // The cross product of all vector pairs corresponding to ordered
    // consecutive segments has to be positive.
    auto crossp = [&](Real2 const& v1, Real2 const& v2) {
        return v1[0] * v2[1] - v1[1] * v2[0];
    };

    // Use the cross_product(last, first) as sign reference
    auto num_corners = corners.size();
    Real2 vec1 = corners[0] - corners[num_corners - 1];
    Real2 vec2 = corners[1] - corners[0];
    real_type const ref = crossp(vec1, vec2);
    for (auto i : range(num_corners - 1))
    {
        vec1 = vec2;
        vec2 = corners[(i + 2) % num_corners] - corners[(i + 1) % num_corners];
        auto val = crossp(vec1, vec2);
        // Make sure the sign is the same as the reference
        if (val * ref < 0 || (!degen_ok && val == 0))
        {
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
