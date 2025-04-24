//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
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
/*!
 * Functor for calculating orientation with a tolerance for collinearity.
 */
template<class T>
class SoftOrientation
{
  public:
    using real_type = T;
    using Real2 = Array<real_type, 2>;

    // Construct with default tolerance
    CELER_CONSTEXPR_FUNCTION SoftOrientation() : soft_zero_{} {}

    // Construct with specified absolute tolerance
    explicit CELER_FUNCTION SoftOrientation(real_type abs_tol)
        : soft_zero_{abs_tol}
    {
    }

    // Calculate orientation with tolerance for collinearity
    CELER_FUNCTION Orientation operator()(Real2 const& a,
                                          Real2 const& b,
                                          Real2 const& c) const
    {
        auto crossp = (b[0] - a[0]) * (c[1] - b[1])
                      - (b[1] - a[1]) * (c[0] - b[0]);
        return soft_zero_(crossp) ? Orientation::collinear
               : crossp < 0       ? Orientation::clockwise
                                  : Orientation::counterclockwise;
    }

  private:
    SoftZero<real_type> soft_zero_;
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
inline bool has_orientation(Span<Real2 const> corners, Orientation o)
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
 * Whether the orientation is the same or degenerate if allowed.
 */
inline bool
is_same_orientation(Orientation a, Orientation b, bool degen_ok = false)
{
    if (a == Orientation::collinear || b == Orientation::collinear)
    {
        return degen_ok;
    }
    return (a == b);
}

//---------------------------------------------------------------------------//
/*!
 * Check if a 2D polygon is convex.
 *
 * \param corners the vertices of the polygon
 * \param degen_ok allow consecutive collinear points
 */
inline bool is_convex(Span<Real2 const> corners, bool degen_ok = false)
{
    CELER_EXPECT(corners.size() > 2);
    auto ref = Orientation::collinear;
    for (auto i : range<size_type>(corners.size()))
    {
        auto j = (i + 1) % corners.size();
        auto k = (i + 2) % corners.size();
        auto cur = calc_orientation(corners[i], corners[j], corners[k]);
        if (ref == Orientation::collinear)
        {
            // First non-collinear point
            ref = cur;
        }
        if (!is_same_orientation(cur, ref, degen_ok))
        {
            // Prohibited collinear orientation, or different orientation from
            // reference
            return false;
        }
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
