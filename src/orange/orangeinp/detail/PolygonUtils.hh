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
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Find orientation of ordered vertices in 2D coordinates.
 */
template<class T>
inline Orientation calc_orientation(celeritas::Array<T, 2> const& a,
                                    celeritas::Array<T, 2> const& b,
                                    celeritas::Array<T, 2> const& c)
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
        if (calc_orientation<Real2::value_type>(
                corners[i], corners[j], corners[k])
            != o)
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
 * Functor for calculating orientation with a tolerance for collinearity.
 *
 * Collinearity is based on a supplied absolute tolerance. For three ordered
 * points a, b, c, point b is collinear if the displacement, d, is less than
 * the absolute tolerance.
 * \verbatim
                b
               . .
             .  .  .
           .    .    .
         .      . d    .
       .  t     .        .
     a . . . . . . . . . . c
   \endverbatim
 * The displacement is calculated as follows.
 *
 * Let:
 * \verbatim
   u = b - a
   v = c - a
   \endverbatim
 *
 * In 2D, the cross product can be written as,
 *
 * \verbatim
   u x v = |u| |v| sin(t),
   \endverbatim
 *
 * noting that this is a different cross product (different vectors) compared
 * to the cross product used for orientation determination. Geometrically, the
 * displacement can be calculated as,
 *
 *  \verbatim
    d = |u| sin(t).
    \endverbatim
 *
 * Therefore,
 *
 * \verbatim
   d = |u| (u x v) / (|u| |v|)
     = (u x v)/|v|.
   \endverbatim
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
        Real2 u{b[0] - a[0], b[1] - a[1]};
        Real2 v{c[0] - a[0], c[1] - a[1]};

        auto cross_product = (u[0] * v[1] - v[0] * u[1]);
        auto magnitude = std::hypot(v[0], v[1]);

        if (magnitude == 0 || soft_zero_(cross_product / magnitude))
        {
            return Orientation::collinear;
        }
        else
        {
            return calc_orientation<real_type>(a, b, c);
        }
    }

  private:
    SoftZero<real_type> soft_zero_;
};

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
        auto cur = calc_orientation<Real2::value_type>(
            corners[i], corners[j], corners[k]);
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
/*!
 * Return the non-collinear subset of the supplied corners, as a copy.
 *
 * Points are checked for collinearity dynamically, i.e, if a point is found to
 * be collinear, it is not used for future collinearity checks.
 */
inline std::vector<Real2>
filter_collinear_points(std::vector<Real2> const& corners, double abs_tol)
{
    CELER_EXPECT(corners.size() >= 3);

    std::vector<Real2> result;
    result.reserve(corners.size());

    SoftOrientation<Real2::value_type> soft_ori(abs_tol);

    // Temporarily assume first point is not collinear. This is necessary to
    // handle the case where all points are locally collinear, but some points
    // are globally non-collinear, e.g., a many-sided regular polygon.
    result.push_back(corners[0]);

    for (auto i : range<size_type>(1, corners.size()))
    {
        auto j = i + 1 < corners.size() ? i + 1 : 0;

        if (soft_ori(result.back(), corners[i], corners[j])
            != Orientation::collinear)
        {
            result.push_back(corners[i]);
        }
    }

    // Make sure there are enough filtered points to specificy a polygon.
    CELER_ASSERT(result.size() >= 3);

    // If it turns out that the first point is actually collinear, remove it.
    if (soft_ori(result.back(), result[0], result[1]) == Orientation::collinear)
    {
        result.erase(result.begin());
    }

    // It shouldn't be possible for the potential removal of the first point
    // to leave us with fewer than 3 points, but check just in case.
    CELER_ENSURE(result.size() >= 3);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
