//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Toroid.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Toroid.hh"

#include "corecel/cont/Array.hh"
#include "corecel/io/StreamToString.hh"
#include "corecel/math/ArrayOperators.hh"
#include "orange/surf/Toroid.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using Real3 = Toroid::Real3;
using Intersections = Toroid::Intersections;

//---------------------------------------------------------------------------//
/*!
 * Fills a list of fewer than 4 roots with "no real positive root"
 */
Intersections make_inters(std::initializer_list<real_type> const& inp)
{
    CELER_EXPECT(inp.size() <= Intersections{}.size());
    Intersections result;
    auto iter = std::copy(inp.begin(), inp.end(), result.begin());
    std::fill(iter, result.end(), NumericLimits<real_type>::infinity());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sorts a given array of four roots and returns the array.
 */
Intersections sorted(Intersections four_roots)
{
    sort(four_roots.begin(), four_roots.end());
    return four_roots;
}

//---------------------------------------------------------------------------//
// TEST CASES
//---------------------------------------------------------------------------//

/*!
 * Test constructors and span output of toroid class
 */
TEST(ToroidTest, construction)
{
    // Position at 1, 2, 3, major rad 10, xy rad 4, z rad 5
    auto check_props = [](Toroid const& tor) {
        Real3 expect{1, 2, 3};
        Real3 actual = tor.origin();
        EXPECT_VEC_EQ(expect, actual);
        EXPECT_EQ(10, tor.major_radius());
        EXPECT_EQ(4, tor.ellipse_xy_radius());
        EXPECT_EQ(5, tor.ellipse_z_radius());
    };

    Toroid tor{Real3{1, 2, 3}, 10, 4, 5};
    check_props(tor);

    {
        // Reconstruction from data
        SCOPED_TRACE("reconstructed");
        Toroid recon{tor.data()};
        check_props(tor);
    }
}

/*!
 * Test sense calculation
 */
TEST(ToroidTest, sense)
{
    Real3 origin{1, 2, 3};
    Toroid tor{origin, 5, 1, 2};
    Real3 inner_points[] = {{5, 0, 0}, {0, 5, 0}, {5 * 0.707, 5 * 0.707, 1.9}};
    for (Real3 const& point : inner_points)
    {
        SCOPED_TRACE("Inner point: " + stream_to_string(point));
        EXPECT_EQ(SignedSense::inside, tor.calc_sense(point + origin));
    }

    static Real3 const outer_points[] = {
        {0, 0, 0},
        {0, 3.9, 0},
        {3.9, 0, 0},
        {-3.9, 0, 0},
        {5, 0, 2.1},
        {6.1, 0, 0},
    };
    for (Real3 const& point : outer_points)
    {
        SCOPED_TRACE("Outer point: " + stream_to_string(point));
        EXPECT_EQ(SignedSense::outside, tor.calc_sense(point + origin));
    }

    static Real3 const edge_points[] = {
        {5.0, 0, 2.0},
        {4.0, 0, 0},
        {6.0, 0, 0},
    };
    for (Real3 const& point : edge_points)
    {
        SCOPED_TRACE("Edge point: " + stream_to_string(point));
        EXPECT_EQ(SignedSense::on, tor.calc_sense(point + origin));
    }
}

/*!
 * Test normal vector calculation
 */
TEST(ToroidTest, normal)
{
    Real3 origin{1, 2, 3};
    Toroid tor{origin, 5, 1, 2};
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1}),
                       tor.calc_normal(origin + Real3{5, 0, 2}));
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, -1}),
                       tor.calc_normal(origin + Real3{5, 0, -2}));
    EXPECT_VEC_SOFT_EQ((Real3{1, 0, 0}),
                       tor.calc_normal(origin + Real3{6, 0, 0}));
    EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}),
                       tor.calc_normal(origin + Real3{4, 0, 0}));
    EXPECT_VEC_SOFT_EQ((Real3{0, 1, 0}),
                       tor.calc_normal(origin + Real3{0, 6, 0}));
}

/*!
 * Test intersection calculation
 */
TEST(ToroidTest, intersect)
{
    Real3 origin{1, 2, 3};
    Toroid tor{origin, 5, 1, 2};

    constexpr SurfaceState off = SurfaceState::off;

    // Ray through center shouldn't hit
    Real3 s{origin + Real3{0, 0, 2}};
    Real3 u{0, 0, -1};
    EXPECT_VEC_SOFT_EQ(make_inters({}),
                       sorted(tor.calc_intersections(s, u, off)));

    // Ray inside and out from center should hit exactly once
    s = origin + Real3{0, 5, 0};
    u = {0, 1, 0};
    EXPECT_VEC_SOFT_EQ(make_inters({1}),
                       sorted(tor.calc_intersections(s, u, off)));

    // Ray inside towards center should hit 3 times
    s = origin + Real3{0, 5, 0};
    u = {0, -1, 0};
    EXPECT_VEC_SOFT_EQ(make_inters({1, 9, 11}),
                       sorted(tor.calc_intersections(s, u, off)));

    // Ray inside towards center again, to have one test that's not nice even
    // numbers
    s = origin + Real3{0.2, 5.1, 0.1};
    u = {-0.039178047638066676, -0.9990402147707002, 0.019589023819033338};
    Intersections expected = sorted(make_inters(
        {1.1022820700552722, 11.093380864669665, 9.1154137268987121}));
    EXPECT_VEC_NEAR(expected,
                    sorted(tor.calc_intersections(s, u, off)),
                    1e-4);  // Not a precision test, that comes later

    // Ray above torus shouldn't hit torus below it
    s = origin + Real3{0.2, 5.1, 2.1};
    u = {1.0 / 9, 4.0 / 9, 8.0 / 9};
    EXPECT_VEC_SOFT_EQ(make_inters({}),
                       sorted(tor.calc_intersections(s, u, off)));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
