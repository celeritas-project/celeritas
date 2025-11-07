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
    EXPECT_EQ(ccw, calc_orientation<real_type>({0, 0}, {4, 4}, {1, 2}));
    EXPECT_EQ(cw, calc_orientation<real_type>({0, 0}, {4, 4}, {2, 1}));
    EXPECT_EQ(col, calc_orientation<real_type>({0, 0}, {4, 4}, {2, 2}));
    EXPECT_EQ(col, calc_orientation<real_type>({0, 0}, {1, 1}, {2, 2}));
    EXPECT_EQ(col, calc_orientation<real_type>({2, 2}, {1, 1}, {0, 0}));
    EXPECT_EQ(col, calc_orientation<real_type>({0, 0}, {0, 0}, {1, 1}));
    EXPECT_EQ(col, calc_orientation<real_type>({0, 0}, {0, 0}, {0, 0}));
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

TEST(PolygonUtilsTest, soft_orientation)
{
    SoftOrientation tight_soft_ori(1e-10);
    SoftOrientation loose_soft_ori(0.01);

    // Basic tests
    EXPECT_EQ(ccw, tight_soft_ori({0, 0}, {4, 4}, {1, 2}));
    EXPECT_EQ(cw, tight_soft_ori({0, 0}, {4, 4}, {2, 1}));
    EXPECT_EQ(col, tight_soft_ori({0, 0}, {4, 4}, {2, 2}));
    EXPECT_EQ(col, tight_soft_ori({0, 0}, {1, 1}, {2, 2}));
    EXPECT_EQ(col, tight_soft_ori({2, 2}, {1, 1}, {0, 0}));
    EXPECT_EQ(col, tight_soft_ori({0, 0}, {0, 0}, {1, 1}));
    EXPECT_EQ(col, tight_soft_ori({0, 0}, {0, 0}, {0, 0}));

    // Collinearity tests
    EXPECT_EQ(cw, tight_soft_ori({0, 0}, {1, 0.009}, {2, 0}));
    EXPECT_EQ(ccw, tight_soft_ori({0, 0}, {1, -0.009}, {2, 0}));

    EXPECT_EQ(col, loose_soft_ori({0, 0}, {1, 0.009}, {2, 0}));
    EXPECT_EQ(col, loose_soft_ori({0, 0}, {1, -0.009}, {2, 0}));

    EXPECT_EQ(cw, loose_soft_ori({0, 0}, {1, 0.011}, {2, 0}));
    EXPECT_EQ(ccw, loose_soft_ori({0, 0}, {1, -0.011}, {2, 0}));
}

TEST(PolygonUtilsTest, convex)
{
    static Real2 const cw_points[] = {{1, 1}, {1, 2}, {2, 2}, {2, 1}};
    EXPECT_TRUE(is_convex(make_span(cw_points)));

    static Real2 const ccw_points[] = {{1, 1}, {2, 1}, {2, 2}, {1, 2}};
    EXPECT_TRUE(is_convex(ccw_points));

    VecReal2 oct(8);
    for (auto i : range(oct.size()))
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

/* Test removal of collinear points using points a through g, which when
 * traversed clockewise form a convex polygon.
 *
 *     c . . . . . d  |_ 2
 *     .              |.
 *     .              |     .
 *    .               |         .
 *    .               |_ 1          e
 *    .               |           .
 *   .                |          .
 *   .                |         .
 *   b________________a____g__f_________
 *   |        |       |       |        |
 *  -1       -0.5     0      0.5       1
 */
TEST(PolygonUtilsTest, filter_collinear_points)
{
    // Point locations, as labeled above
    Real2 a = {0, 0};
    Real2 b = {-1, -1e-5};
    Real2 c = {-0.9, -0.1};
    Real2 d = {0.75, 1};
    Real2 e = {0.75, 0.5};
    Real2 f = {0.5, 1e-5};
    Real2 g = {0.35, 1e-6};

    using VecReal2 = std::vector<Real2>;
    real_type tol = 0.01;

    // No collinear points (b through f)
    VecReal2 points{b, c, d, e, f};
    VecReal2 exp = points;
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Point a is collinear, using a through f, and a comes first
    points = {a, b, c, d, e, f};
    exp = {b, c, d, e, f};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes second
    points = {f, a, b, c, d, e};
    exp = {f, b, c, d, e};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes third
    points = {e, f, a, b, c, d};
    exp = {e, f, b, c, d};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes last
    points = {b, c, d, e, f, a};
    exp = {b, c, d, e, f};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes second
    points = {f, a, b, c, d, e};
    exp = {f, b, c, d, e};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Points a and g are collinear, using a through g, and a comes first
    points = {a, b, c, d, e, f, g};
    exp = {b, c, d, e, f};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes second
    points = {g, a, b, c, d, e, f};
    exp = {b, c, d, e, f};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes third
    points = {f, g, a, b, c, d, e};
    exp = {f, b, c, d, e};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes second to last
    points = {c, d, e, f, g, a, b};
    exp = {c, d, e, f, b};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));

    // Same, but a comes last
    points = {b, c, d, e, f, g, a};
    exp = {b, c, d, e, f};
    EXPECT_EQ(exp, filter_collinear_points(points, tol));
}

/* Test pathological case consisting of a many-sided regular polygon with every
 * point soft collinear with its neighbors due to a large tolerance.
 */
TEST(PolygonUtilsTest, filter_collinear_points_pathological)
{
    // Create a many-sided regular polygon by placing 20 equally spaced points
    // on a circle of radius = 1, in clockwise order.
    VecReal2 points(20);
    real_type step = static_cast<real_type>(2 * constants::pi / points.size());

    for (auto i : range(points.size()))
    {
        real_type theta = -step * i;
        points[i] = {std::cos(theta), std::sin(theta)};
    }

    // Choose a tolerance such that adjacent points are soft collinear
    real_type tol = 0.1;
    SoftOrientation soft_ori(tol);
    EXPECT_EQ(Orientation::collinear,
              soft_ori(points[0], points[1], points[2]));

    // Check that filtering provides more than zero points, in this case 7
    auto filtered_points = filter_collinear_points(points, tol);
    EXPECT_EQ(7, filtered_points.size());
}

TEST(PolygonUtilsTest, calc_extrema)
{
    static Real2 const polygon[] = {
        {2, -3.5}, {0.1, -3.8}, {-5.03, 0.3}, {-1, 5.8}, {10.11, 9.1}, {6, 5.3}};
    auto [x_min, x_max] = find_extrema(make_span(polygon), 0);
    auto [y_min, y_max] = find_extrema(make_span(polygon), 1);

    EXPECT_SOFT_EQ(-5.03, x_min);
    EXPECT_SOFT_EQ(10.11, x_max);
    EXPECT_SOFT_EQ(-3.8, y_min);
    EXPECT_SOFT_EQ(9.1, y_max);
}

TEST(PolygonUtilsTest, normal_from_triangle)
{
    constexpr auto dir = real_type{1} / constants::sqrt_three;

    // Construct from three points, in this case a plane passing through the
    // point (1, 2, 3) with slope (1, 1, 1). Specifying the points in clockwise
    // order gives a negative normal.
    EXPECT_VEC_SOFT_EQ((Real3{-dir, -dir, -dir}),
                       normal_from_triangle({2, 1, 3}, {-3, 5, 4}, {4, 7, -5}));

    // Specifying the points in counterclockwise order flips the and normal
    EXPECT_VEC_SOFT_EQ((Real3{dir, dir, dir}),
                       normal_from_triangle({4, 7, -5}, {-3, 5, 4}, {2, 1, 3}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
