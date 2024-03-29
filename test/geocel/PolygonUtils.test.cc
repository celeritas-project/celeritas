//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/PolygonUtils.test.cc
//---------------------------------------------------------------------------//
#include "geocel/detail/PolygonUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using Real2 = Array<real_type, 2>;
using VecReal2 = std::vector<Real2>;
using Real3 = Array<real_type, 3>;
using celeritas::axpy;
using namespace celeritas::geoutils;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(PolygonUtilsTest, orient_2D)
{
    EXPECT_TRUE(orientation_2D(Real2{0, 0}, Real2{4, 4}, Real2{1, 2}) == kCCW);
    EXPECT_TRUE(orientation_2D(Real2{0, 0}, Real2{4, 4}, Real2{2, 1}) == kCW);
    EXPECT_TRUE(orientation_2D(Real2{0, 0}, Real2{4, 4}, Real2{2, 2}) == kLine);
}

TEST(PolygonUtilsTest, convexity)
{
    VecReal2 cw{{1, 1}, {1, 2}, {2, 2}, {2, 1}};
    EXPECT_TRUE(is_convex_2D(cw));

    VecReal2 ccw{{1, 1}, {2, 1}, {2, 2}, {1, 2}};
    EXPECT_TRUE(is_convex_2D(ccw));

    VecReal2 oct{8};
    for (size_type i = 0; i < 8; ++i)
    {
        oct[i] = {std::cos(2 * m_pi * i / 8), std::sin(2 * m_pi * i / 8)};
    }
    EXPECT_TRUE(is_convex_2D(oct));

    // not properly ordered
    VecReal2 bad{{1, 1}, {2, 2}, {2, 1}, {1, 2}};
    EXPECT_FALSE(is_convex_2D(bad));
}

TEST(PolygonUtilsTest, degenerate)
{
    // degenerate: all points are colinear
    VecReal2 line{{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    EXPECT_FALSE(is_convex_2D(line));

    // only three points are collinear
    VecReal2 degen{{1, 1}, {2, 2}, {3, 3}, {2, 4}};
    bool degen_ok = true;
    EXPECT_FALSE(is_convex_2D(degen));  // default: degen_ok = false
    EXPECT_TRUE(is_convex_2D(degen, degen_ok));

    // degenerate: repeated consecutive points
    VecReal2 repeated{{0, 0}, {1, 0}, {1, 1}, {0.5, 0.5}, {0.5, 0.5}, {0, 1}};
    EXPECT_FALSE(is_convex_2D(repeated));
}

TEST(PolygonUtilsTest, self_intersect)
{
    VecReal2 self_int{{0, 0}, {1, 1}, {1, 0}, {0, 1}};
    EXPECT_FALSE(is_convex_2D(self_int));

    VecReal2 self_int2{{0, 0}, {1, 1}, {0, 1}, {1, 0}};
    EXPECT_FALSE(is_convex_2D(self_int2));
}

TEST(PolygonUtilsTest, planar_3D)
{
    Real3 a{-2, 2, -2}, b{2, 2, -2}, c{2, 2, 2}, d{-2, 2, 2};
    EXPECT_TRUE(is_planar_3D(a, b, c, d));
    EXPECT_TRUE(is_planar_3D(a, b, d, c));  // proper ordering not required
    EXPECT_FALSE(is_planar_3D(a, b, c, Real3{0, 0, 0}));
}

TEST(PolygonUtilsTest, planar_tolerance)
{
    double eps_lo = 1e-8;
    Real3 a{-2, 2, -2}, b{2, 2, -2}, c{2, 2, 2}, d{-2, 2, 2};
    Real3 dy{0, 1, 0};

    // effect on reference corner
    Real3 aa{a};
    axpy(eps_lo, dy, &aa);
    EXPECT_TRUE(is_planar_3D(aa, b, c, d));
    axpy(10 * eps_lo, dy, &aa);
    EXPECT_FALSE(is_planar_3D(aa, b, c, d));

    // effect on non-ref corner
    Real3 bb{b};
    axpy(eps_lo, dy, &bb);
    EXPECT_TRUE(is_planar_3D(a, bb, c, d));
    axpy(10 * eps_lo, dy, &bb);
    EXPECT_FALSE(is_planar_3D(a, bb, c, d));

    // effect on test corner
    Real3 dd{d};
    axpy(eps_lo, dy, &dd);
    EXPECT_TRUE(is_planar_3D(a, b, c, dd));
    axpy(10 * eps_lo, dy, &dd);
    EXPECT_FALSE(is_planar_3D(a, b, c, dd));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
