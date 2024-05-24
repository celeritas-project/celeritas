//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orangeinp/detail/PolygonUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/PolygonUtils.hh"

#include <vector>

#include "corecel/Constants.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

using Real2 = Array<real_type, 2>;
using VecReal2 = std::vector<Real2>;
using constants::pi;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(PolygonUtilsTest, calc_orientation)
{
    EXPECT_TRUE(calc_orientation(Real2{0, 0}, Real2{4, 4}, Real2{1, 2})
                == Orientation::counterclockwise);
    EXPECT_TRUE(calc_orientation(Real2{0, 0}, Real2{4, 4}, Real2{2, 1})
                == Orientation::clockwise);
    EXPECT_TRUE(calc_orientation(Real2{0, 0}, Real2{4, 4}, Real2{2, 2})
                == Orientation::collinear);
}

TEST(PolygonUtilsTest, has_orientation)
{
    EXPECT_TRUE(has_orientation(
        make_span(VecReal2{{-19, -30}, {-19, 30}, {21, 30}, {21, -30}}),
        Orientation::clockwise));
    EXPECT_FALSE(has_orientation(
        make_span(VecReal2{{-19, -30}, {-19, 30}, {21, 30}, {21, -30}}),
        Orientation::counterclockwise));

    EXPECT_TRUE(has_orientation(
        make_span(VecReal2{{-2, -2}, {0, -2}, {0, 0}, {-2, 0}}),
        Orientation::counterclockwise));
}

TEST(PolygonUtilsTest, convexity)
{
    VecReal2 cw{{1, 1}, {1, 2}, {2, 2}, {2, 1}};
    EXPECT_TRUE(is_convex(make_span(cw)));

    Real2 ccw[] = {{1, 1}, {2, 1}, {2, 2}, {1, 2}};
    EXPECT_TRUE(is_convex(ccw));

    VecReal2 oct{8};
    for (size_type i = 0; i < 8; ++i)
    {
        oct[i] = {std::cos(2 * pi * i / 8), std::sin(2 * pi * i / 8)};
    }
    EXPECT_TRUE(is_convex(make_span(oct)));

    // not properly ordered
    Real2 bad[] = {{1, 1}, {2, 2}, {2, 1}, {1, 2}};
    EXPECT_FALSE(is_convex(bad));
}

TEST(PolygonUtilsTest, degenerate)
{
    // degenerate: all points are colinear
    Real2 line[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    EXPECT_FALSE(is_convex(line));

    // only three points are collinear
    Real2 degen[] = {{1, 1}, {2, 2}, {3, 3}, {2, 4}};
    EXPECT_FALSE(is_convex(degen));
    EXPECT_TRUE(is_convex(degen, /* degen_ok = */ true));

    // degenerate: repeated consecutive points
    Real2 repeated[] = {{0, 0}, {1, 0}, {1, 1}, {0.5, 0.5}, {0.5, 0.5}, {0, 1}};
    EXPECT_FALSE(is_convex(repeated));
}

TEST(PolygonUtilsTest, self_intersect)
{
    Real2 self_int[] = {{0, 0}, {1, 1}, {1, 0}, {0, 1}};
    EXPECT_FALSE(is_convex(self_int));

    Real2 self_int2[] = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
    EXPECT_FALSE(is_convex(self_int2));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
