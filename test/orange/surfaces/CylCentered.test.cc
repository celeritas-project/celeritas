//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CylCentered.test.cc
//---------------------------------------------------------------------------//
#include "orange/surfaces/CylCentered.hh"

#include <algorithm>
#include <cmath>
#include <vector>
#include "base/Algorithms.hh"
#include "celeritas_test.hh"

using celeritas::CCylX;
using celeritas::CCylY;
using celeritas::CCylZ;
using celeritas::no_intersection;
using celeritas::SignedSense;
using celeritas::SurfaceState;

using celeritas::ipow;
using celeritas::Real3;
using celeritas::real_type;

using Intersections = CCylX::Intersections;
using VecReal       = std::vector<real_type>;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST(TestCCylX, construction)
{
    EXPECT_EQ(1, CCylX::Storage::extent);
    EXPECT_EQ(2, CCylX::Intersections{}.size());

    CCylX c(4.0);

    const real_type expected_data[] = {ipow<2>(4)};

    EXPECT_VEC_SOFT_EQ(expected_data, c.data());
}

TEST(TestCCylX, sense)
{
    CCylX cyl(4.0);

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense(Real3{0, 3, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense(Real3{0, 0, 9}));
}

TEST(TestCCylX, normal)
{
    CCylX cyl(3.45);

    EXPECT_VEC_SOFT_EQ((Real3{0, 1, 0}), cyl.calc_normal(Real3{1.23, 3.45, 0}));
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1}),
                       cyl.calc_normal(Real3{0.0, 0.0, 4.45}));

    // Normal from center of cylinder is ill-defined but shouldn't raise an
    // error
    auto norm = cyl.calc_normal(Real3{10.0, 0.0, 0.0});
    EXPECT_TRUE(std::isnan(norm[0]));
    EXPECT_TRUE(std::isnan(norm[1]));
    EXPECT_TRUE(std::isnan(norm[2]));
}

TEST(TestCCylX, intersect)
{
    Intersections distances{-1, -1};

    // From inside
    CCylX cyl(3.0);
    distances = cyl.calc_intersections(
        Real3{0, 0, 1.5}, Real3{0, 1, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.598076211353316, distances[1]);
    EXPECT_EQ(no_intersection(), distances[0]);

    // From outside, hitting both
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{0, 0, -5.0}, Real3{0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_SOFT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{0, -3, -5.0}, Real3{0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{0, 0, -5.0}, Real3{0, 1, 0}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST(TestCCylX, intersect_from_surface)
{
    Intersections distances;

    CCylX cyl(3.45);

    // One intercept

    distances = cyl.calc_intersections(
        Real3{1.23, 3.45, 0.0}, Real3{0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(6.9, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // No intercepts
    distances = cyl.calc_intersections(
        Real3{1.23, 4.68, 2.34}, Real3{0, 1, 0}, SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

//---------------------------------------------------------------------------//

TEST(TestCCylY, sense)
{
    CCylY cyl(3.0);

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense(Real3{1.5, 0, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense(Real3{3.01, 0, 0}));
}

TEST(TestCCylY, intersect)
{
    CCylY::Intersections distances{-1, -1};

    // From inside
    CCylY cyl(3.0);
    distances = cyl.calc_intersections(
        Real3{1.5, 0, 0}, Real3{0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.598076211353316, distances[1]);
    EXPECT_EQ(no_intersection(), distances[0]);

    // From outside, hitting both
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{-5.0, 0, 0}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_SOFT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{-5.0, 0, -3}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{-5.0, 0, 0}, Real3{0, 0, 1}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST(TestCCylY, intersect_from_surface)
{
    CCylY::Intersections distances;

    CCylY cyl(3.45);

    // One intercept

    distances = cyl.calc_intersections(
        Real3{3.45, 1.23, 0}, Real3{-1, 0, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(6.9, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances = cyl.calc_intersections(
        Real3{1.5528869044748548, 12345., 3.08577380894971},
        Real3{0, 0.6, -0.8},
        SurfaceState::on);
    EXPECT_SOFT_EQ(7.714434522374273, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // No intercepts

    distances = cyl.calc_intersections(
        Real3{3.45, 1.23, 0.0}, Real3{1, 0, 0}, SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances = cyl.calc_intersections(
        Real3{1.5528869044748548, 12345., 3.08577380894971},
        Real3{0, 0.6, 0.8},
        SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

//---------------------------------------------------------------------------//

TEST(TestCCylZ, sense)
{
    CCylZ cyl(3.0);

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense(Real3{1.5, 0, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense(Real3{3.01, 0, 0}));
}

TEST(TestCCylZ, calc_intersections)
{
    Intersections distances{-1, -1};

    // From inside
    CCylZ cyl(3.0);
    distances = cyl.calc_intersections(
        Real3{1.5, 0, 0}, Real3{0, 1, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.598076211353316, distances[1]);
    EXPECT_EQ(no_intersection(), distances[0]);

    // From outside, hitting both
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{-5.0, 0, 0}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{-5.0, -3, 0}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances                   = cyl.calc_intersections(
        Real3{-5.0, 0, 0}, Real3{0, 1, 0}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST(TestCCylZ, calc_intersections_on_surface)
{
    CCylZ::Intersections distances;
    const real_type      eps = 1.e-4;

    {
        CCylZ cyl(1.0);

        // Heading toward, slightly inside
        distances = cyl.calc_intersections(
            Real3{-1 + eps, 0, 0}, Real3{1, 0, 0}, SurfaceState::on);
        EXPECT_SOFT_NEAR(2.0 - eps, distances[0], eps);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Heading away, slightly inside
        distances = cyl.calc_intersections(
            Real3{-1 + eps, 0, 0}, Real3{-1, 0, 0}, SurfaceState::on);
        EXPECT_EQ(no_intersection(), distances[0]);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Tangent
        distances = cyl.calc_intersections(
            Real3{-1 + eps, 0, 0}, Real3{0, 1, 0}, SurfaceState::on);
        EXPECT_EQ(no_intersection(), distances[0]);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Heading in, slightly inside
        distances = cyl.calc_intersections(
            Real3{-1 + eps, 0, 0}, Real3{1, 0, 0}, SurfaceState::on);
        EXPECT_SOFT_NEAR(2.0 - eps, distances[0], eps);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Heading away, slightly outside
        distances = cyl.calc_intersections(
            Real3{1.0 - eps, 0, 0}, Real3{1, 0, 0}, SurfaceState::on);
        EXPECT_EQ(no_intersection(), distances[0]);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Tangent, slightly inside
        distances = cyl.calc_intersections(
            Real3{-1 + eps, 0, 0}, Real3{0, 1, 0}, SurfaceState::on);
        EXPECT_EQ(no_intersection(), distances[0]);
        EXPECT_EQ(no_intersection(), distances[1]);
    }
}

// Fused multiply-add on some CPUs in opt mode can cause the results of
// nearly-tangent cylinder checking to change.
TEST(TestCCylZ, multi_intersect)
{
    constexpr int Y = static_cast<int>(celeritas::Axis::y);

    CCylZ cyl(10.0);

    VecReal all_first_distances;
    VecReal all_distances;
    VecReal all_y;
    for (real_type x : {-10 + 1e-7, -1e-7, -0.0, 0.0, 1e-7, 10 - 1e-7})
    {
        for (real_type v : {-1.0, 1.0})
        {
            Real3 pos{x, -10.0001 * v, 0};
            Real3 dir{0, v, 0};

            Intersections d;

            // Transport to inside of cylinder
            d = cyl.calc_intersections(pos, dir, SurfaceState::off);
            ASSERT_NE(no_intersection(), d[0]);
            all_first_distances.push_back(d[0]);
            pos[Y] += d[0] * dir[Y];
            all_y.push_back(pos[Y]);

            // Transport to other side of cylinder
            d = cyl.calc_intersections(pos, dir, SurfaceState::on);
            all_distances.push_back(d[0]);
            ASSERT_NE(no_intersection(), d[0]);
            pos[Y] += d[0] * dir[Y];

            // We're done
            d = cyl.calc_intersections(pos, dir, SurfaceState::on);
            EXPECT_EQ(no_intersection(), d[0]);
        }
    }

    const real_type expected_all_first_distances[] = {9.99869,
                                                      9.99869,
                                                      0.0001,
                                                      0.0001,
                                                      0.0001,
                                                      0.0001,
                                                      0.0001,
                                                      0.0001,
                                                      0.0001,
                                                      0.0001,
                                                      9.99869,
                                                      9.99869};
    EXPECT_VEC_NEAR(expected_all_first_distances, all_first_distances, 1e-5);

    const real_type expected_all_y[] = {0.00141421,
                                        -0.00141421,
                                        10,
                                        -10,
                                        10,
                                        -10,
                                        10,
                                        -10,
                                        10,
                                        -10,
                                        0.00141421,
                                        -0.00141421};
    EXPECT_VEC_NEAR(expected_all_y, all_y, 1e-5);

    const real_type expected_all_distances[] = {0.00282843,
                                                0.00282843,
                                                20,
                                                20,
                                                20,
                                                20,
                                                20,
                                                20,
                                                20,
                                                20,
                                                0.00282843,
                                                0.00282843};
    EXPECT_VEC_NEAR(expected_all_distances, all_distances, 1e-5);
}

//---------------------------------------------------------------------------//
/*!
 * Test initialization on or near boundary
 */
class DegenerateBoundaryTest : public celeritas::Test
{
  protected:
    void run(real_type xdir) const;

    void run_all()
    {
        for (real_type r : {0.93, 1.0})
        {
            radius = r;
            for (real_type dir : {1, -1})
            {
                std::ostringstream msg;
                msg << "r=" << radius << ", dir_x=" << dir;
                SCOPED_TRACE(msg.str());

                run(1);
                run(-1);
            }
        }
    }

  protected:
    real_type radius = -1;
    real_type eps    = -1;
};

void DegenerateBoundaryTest::run(real_type xdir) const
{
    CCylZ                cyl(radius);
    CCylZ::Intersections distances = {-1, -1};
    const real_type      tol       = std::max(1.e-14, 2 * std::fabs(eps));

    // Distance across the cylinder
    const real_type diameter = 2 * radius;

    Real3 pos = {0, 0, 0};
    Real3 dir = {xdir, 0, 0};

    //// Inward boundary ////
    pos[0]    = -xdir * (diameter / 2 + eps);
    distances = cyl.calc_intersections(pos, dir, SurfaceState::on);
    EXPECT_SOFT_NEAR(diameter + eps, distances[0], tol);
    EXPECT_EQ(no_intersection(), distances[1]);

    //// Outward boundary ////
    pos[0]    = xdir * (diameter / 2 + eps);
    distances = cyl.calc_intersections(pos, dir, SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST_F(DegenerateBoundaryTest, neg)
{
    eps = -1.0e-8;
    run_all();
}

TEST_F(DegenerateBoundaryTest, DISABLED_zero)
{
    eps = 0.0;
    run_all();
}

TEST_F(DegenerateBoundaryTest, pos)
{
    eps = 1.e-8;
    run_all();
}
