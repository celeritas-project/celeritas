//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Toroid.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Toroid.hh"

#include "corecel/cont/Array.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"
namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using Real3 = Toroid::Real3;
/*!
 * Test constructors and span output of toroid class
 */
TEST(ToroidTest, construction)
{
    // Position at 1, 2, 3, major rad 10, xy rad 4, z rad 5
    auto check_props =
        [](Toroid const& tor) {
            EXPECT_VEC_EQ({1, 2, 3}, tor.origin());
            EXPECT_EQ(10, tor.major_radius());
            EXPECT_EQ(4, tor.ellipse_xy_radius());
            EXPECT_EQ(5, tor.ellipse_z_radius());
        }

    Toroid tor{{1, 2, 3}, 10, 4, 5};
    check_props(tor);

    {
        // Reconstruction from data
        SCOPED_TRACE("reconstructed");
        Toroid recon{tor.data()};
        check_props(tor);
    }
}

Real3 add(Real3 const a, Real3 const b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
/*!
 * Test sense calculation
 */
TEST(ToroidTest, sense)
{
    Real3 origin{1, 2, 3} Toroid tor{origin, 5, 1, 2};
    Real3 inner_points[] = {{5, 0, 0}, {0, 5, 0}, {5 * 0.707, 5 * 0.707, 1.9}};
    for (Real3 const& point : inner_points)
    {
        SCOPED_TRACE("Inner point: " + point.to_string());
        EXPECT_EQ(SignedSense::inside, tor.calc_sense(add(point, origin)));
    }

    Real3 outer_points[] = {{0, 0, 0},
                            {0, 3.9, 0},
                            {3.9, 0, 0},
                            {-3.9, 0, 0},
                            {5, 0, 1.1},
                            {6.1, 0, 0}};
    for (Real3 const& point : outer_points)
    {
        SCOPED_TRACE("Outer point: " + point.to_string());
        EXPECT_EQ(SignedSense::outside, tor.calc_sense(add(point, origin)));
    }

    Real3 edge_points[] = {{5.0, 0, 1.0}, {4.0, 0, 0}, {6.0, 0, 0}};
    for (Real3 const& point : edge_points)
    {
        SCOPED_TRACE("Edge point: " + point.to_string());
        EXPECT_EQ(SignedSense::on, tor.calc_sense(add(point, origin)));
    }
}

/*!
 * Test normal vector calculation
 */
TEST(ToroidTest, normal)
{
    Real3 origin{1, 2, 3};
    Toroid tor{origin, 5, 1, 2};
    EXPECT_VEC_SOFT_EQ(tor.calc_normal(add(origin, {5.0, 0, 1.0})),
                       {0, 0, 1.0});
    EXPECT_VEC_SOFT_EQ(tor.calc_normal(add(origin, {5.0, 0, -1.0})),
                       {0, 0, -1.0});
    EXPECT_VEC_SOFT_EQ(tor.calc_normal(add(origin, {6.0, 0, 0})), {1.0, 0, 0});
    EXPECT_VEC_SOFT_EQ(tor.calc_normal(add(origin, {4.0, 0, 0})), {-1.0, 0, 0});
    EXPECT_VEC_SOFT_EQ(tor.calc_normal(add(origin, {0, 6.0, 0})), {0, 1.0, 0});
    EXPECT_VEC_SOFT_EQ(tor.calc_normal(add(origin, {0, 4.0, 0})), {0, -1.0, 0});
}

/*!
 * Test intersection calculation
 */
TEST(ToroidTest, intersect) {}
}  // namespace test
}  // namespace celeritas
