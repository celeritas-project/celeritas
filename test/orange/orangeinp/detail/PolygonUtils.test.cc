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

TEST(PolygonUtilsTest, planar)
{
    Real3 a{-2, 2, -2}, b{2, 2, -2}, c{2, 2, 2}, d{-2, 2, 2};
    EXPECT_TRUE(is_planar(a, b, c, d));
    EXPECT_TRUE(is_planar(a, b, d, c));  // proper ordering not required
    EXPECT_FALSE(is_planar(a, b, c, Real3{0, 0, 0}));
}

TEST(PolygonUtilsTest, planar_tolerance)
{
    real_type const eps_lo = real_type{0.1} * SoftEqual<>{}.rel();
    real_type const eps_hi = 2 * SoftEqual<>{}.rel();
    Real3 a{-2, 2, -2}, b{2, 2, -2}, c{2, 2, 2}, d{-2, 2, 2};
    Real3 dy{0, 1, 0};

    // effect on reference corner
    Real3 aa{a};
    axpy(eps_lo, dy, &aa);
    EXPECT_TRUE(is_planar(aa, b, c, d));
    axpy(eps_hi, dy, &aa);
    EXPECT_FALSE(is_planar(aa, b, c, d));

    // effect on non-ref corner
    Real3 bb{b};
    axpy(eps_lo, dy, &bb);
    EXPECT_TRUE(is_planar(a, bb, c, d));
    axpy(eps_hi, dy, &bb);
    EXPECT_FALSE(is_planar(a, bb, c, d));

    // effect on test corner
    Real3 dd{d};
    axpy(eps_lo, dy, &dd);
    EXPECT_TRUE(is_planar(a, b, c, dd));
    axpy(eps_hi, dy, &dd);
    EXPECT_FALSE(is_planar(a, b, c, dd));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
