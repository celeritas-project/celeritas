//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/PolygonUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/PolygonUtils.hh"

#include <vector>

#include "corecel/Constants.hh"
#include "corecel/cont/ArrayIO.hh"
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

constexpr auto ccw = Orientation::counterclockwise;
constexpr auto cw = Orientation::clockwise;
constexpr auto col = Orientation::collinear;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(PolygonUtilsTest, calc_orientation)
{
    EXPECT_EQ(ccw, calc_orientation({0, 0}, {4, 4}, {1, 2}));
    EXPECT_EQ(cw, calc_orientation({0, 0}, {4, 4}, {2, 1}));
    EXPECT_EQ(col, calc_orientation({0, 0}, {4, 4}, {2, 2}));
    EXPECT_EQ(col, calc_orientation({0, 0}, {1, 1}, {2, 2}));
    EXPECT_EQ(col, calc_orientation({2, 2}, {1, 1}, {0, 0}));
    EXPECT_EQ(col, calc_orientation({0, 0}, {0, 0}, {1, 1}));
    EXPECT_EQ(col, calc_orientation({0, 0}, {0, 0}, {0, 0}));
}

TEST(PolygonUtilsTest, has_orientation)
{
    static Real2 const cw_points[]
        = {{-19, -30}, {-19, 30}, {21, 30}, {21, -30}};
    EXPECT_TRUE(has_orientation(make_span(cw_points), cw));
    EXPECT_FALSE(has_orientation(make_span(cw_points), ccw));

    static Real2 const ccw_points[] = {{-2, -2}, {0, -2}, {0, 0}, {-2, 0}};
    EXPECT_TRUE(has_orientation(make_span(ccw_points), ccw));
}

TEST(PolygonUtilsTest, is_same_orientation)
{
    EXPECT_TRUE(is_same_orientation(cw, cw));
    EXPECT_FALSE(is_same_orientation(col, col));  // collinear prohibited
    EXPECT_FALSE(is_same_orientation(ccw, cw));
    EXPECT_FALSE(is_same_orientation(cw, col));
    EXPECT_FALSE(is_same_orientation(col, cw));

    constexpr bool degen_ok = true;
    EXPECT_TRUE(is_same_orientation(cw, cw, degen_ok));
    EXPECT_TRUE(is_same_orientation(col, col, degen_ok));
    EXPECT_FALSE(is_same_orientation(ccw, cw, degen_ok));
    EXPECT_TRUE(is_same_orientation(cw, col, degen_ok));
    EXPECT_TRUE(is_same_orientation(col, cw, degen_ok));
}

TEST(PolygonUtilsTest, convex)
{
    static Real2 const cw_points[] = {{1, 1}, {1, 2}, {2, 2}, {2, 1}};
    EXPECT_TRUE(is_convex(make_span(cw_points)));

    static Real2 const ccw_points[] = {{1, 1}, {2, 1}, {2, 2}, {1, 2}};
    EXPECT_TRUE(is_convex(ccw_points));

    VecReal2 oct{8};
    for (size_type i = 0; i < 8; ++i)
    {
        oct[i] = {std::cos(static_cast<real_type>(2 * constants::pi * i / 8)),
                  std::sin(static_cast<real_type>(2 * constants::pi * i / 8))};
    }
    EXPECT_TRUE(is_convex(make_span(oct)));

    // not properly ordered
    Real2 bad[] = {{1, 1}, {2, 2}, {2, 1}, {1, 2}};
    EXPECT_FALSE(is_convex(bad));
}

TEST(PolygonUtilsTest, convex_degenerate)
{
    // degenerate: all points are collinear
    static Real2 const line[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    EXPECT_FALSE(is_convex(line));
    EXPECT_TRUE(is_convex(line, /* degen_ok = */ true));

    // only three points are collinear
    static Real2 const degen[] = {{1, 1}, {2, 2}, {3, 3}, {2, 4}};
    EXPECT_FALSE(is_convex(degen));
    EXPECT_TRUE(is_convex(degen, /* degen_ok = */ true));

    // first and last are collinear
    static Real2 const degen3[] = {{1, 1}, {2, 2}, {0, 2}, {0, 0}};
    EXPECT_FALSE(is_convex(degen3));
    EXPECT_TRUE(is_convex(degen3, /* degen_ok = */ true));

    // degenerate: repeated consecutive points
    static Real2 const repeated[]
        = {{0, 0}, {1, 0}, {1, 1}, {0.5, 0.5}, {0.5, 0.5}, {0, 1}};
    EXPECT_FALSE(is_convex(repeated));
}

TEST(PolygonUtilsTest, convex_self_intersect)
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
