//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/CylCentered.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/CylCentered.hh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using Intersections = CCylX::Intersections;
using VecReal = std::vector<real_type>;

real_type min_intersection(Intersections const& i)
{
    if (i[0] == 0 && i[1] == 0)
        return no_intersection();
    else if (i[0] == 0)
        return i[1];
    else if (i[1] == 0)
        return i[0];
    else if (i[0] < i[1])
        return i[0];
    return i[1];
}

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
    distances = cyl.calc_intersections(
        Real3{0, 0, -5.0}, Real3{0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_SOFT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(
        Real3{0, -3, -5.0}, Real3{0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(
        Real3{0, 0, -5.0}, Real3{0, 1, 0}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From inside, exactly along the axis
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(
        Real3{0.123, 0.345, 0.456}, Real3{1, 0, 0}, SurfaceState::off);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From inside, nearly along the axis
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(Real3{0.123, 0.345, 0.456},
                                       Real3{9.99999999999408140e-01,
                                             4.09517743700767399e-07,
                                             1.00812588415826643e-06},
                                       SurfaceState::off);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
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
    distances = cyl.calc_intersections(
        Real3{-5.0, 0, 0}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_SOFT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(
        Real3{-5.0, 0, -3}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(
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
    distances = cyl.calc_intersections(
        Real3{-5.0, 0, 0}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(
        Real3{-5.0, -3, 0}, Real3{1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances = cyl.calc_intersections(
        Real3{-5.0, 0, 0}, Real3{0, 1, 0}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST(TestCCylZ, calc_intersections_on_surface)
{
    CCylZ::Intersections distances;
    const real_type eps = 1.e-4;

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
TEST(TestCCylZ, multi_tangent_intersect)
{
    constexpr int Y = static_cast<int>(Axis::y);

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

            real_type d;

            // Transport to inside of cylinder
            d = min_intersection(
                cyl.calc_intersections(pos, dir, SurfaceState::off));
            ASSERT_NE(no_intersection(), d);
            all_first_distances.push_back(d);
            pos[Y] += d * dir[Y];
            all_y.push_back(pos[Y]);

            // Transport to other side of cylinder
            d = min_intersection(
                cyl.calc_intersections(pos, dir, SurfaceState::on));
            all_distances.push_back(d);
            if (d == no_intersection())
                continue;

            pos[Y] += d * dir[Y];

            // We're done
            d = min_intersection(
                cyl.calc_intersections(pos, dir, SurfaceState::on));
            EXPECT_EQ(no_intersection(), d);
        }
    }

    // clang-format off
    const real_type expected_all_first_distances[] = {9.99869, 9.99869, 0.0001,
        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 9.99869,
        9.99869};
    EXPECT_VEC_NEAR(expected_all_first_distances, all_first_distances, 1e-5);

    const real_type expected_all_y[] = {0.00141421, -0.00141421, 10, -10, 10,
        -10, 10, -10, 10, -10, 0.00141421, -0.00141421};
    EXPECT_VEC_NEAR(expected_all_y, all_y, 1e-5);

    const real_type expected_all_distances[] = {0.00282843, 0.00282843, 20,
        20, 20, 20, 20, 20, 20, 20, 0.00282843, 0.00282843};
    EXPECT_VEC_NEAR(expected_all_distances, all_distances, 1e-5);
    // clang-format on
}

TEST(TestCCylZ, multi_along_intersect)
{
    CCylZ cyl(30.0);

    VecReal all_first_distances;
    VecReal all_errors;

    for (real_type sqrt_eps : {1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7})
    {
        real_type eps = ipow<2>(sqrt_eps);
        for (Real3 const& orig_pos :
             {Real3{7, 8, 9}, Real3{29, 0, 0}, Real3{30 - eps, 0, 0}})
        {
            for (Real3 dir : {
                     Real3{0, 0, 1},
                     Real3{eps, 0, 1},
                     Real3{sqrt_eps, sqrt_eps, 1},
                     Real3{0, sqrt_eps, 1},
                 })
            {
                Real3 pos = orig_pos;
                normalize_direction(&dir);

                // Transport to inside of cylinder
                real_type d = min_intersection(
                    cyl.calc_intersections(pos, dir, SurfaceState::off));
                all_first_distances.push_back(d);
                if (d == no_intersection())
                    continue;

                axpy(d, dir, &pos);
                real_type boundary_error = std::hypot(pos[0], pos[1])
                                           - real_type(30);
                all_errors.push_back(
                    boundary_error
                    / (std::numeric_limits<real_type>::epsilon() * d));

                EXPECT_LT(std::fabs(boundary_error), real_type(1))
                    << "huge error " << boundary_error << " from " << orig_pos
                    << " along " << dir << " with calculated distance " << d;
            }
        }
    }

    constexpr real_type inf = no_intersection();

    // For long distances we can overshoot or undershoot due to numerical
    // error.

    // clang-format off
    static const double expected_all_first_distances[] = {inf, 2191.4760245467,
        138.43704564716, 212.77500479316, inf, 100.00499987501,
        9.9365248561264, 77.194559393782, inf, 1.00004999875, 0.10097822248343,
        7.7839514386978, inf, 219136.64666597, 1370.868072293, 2117.2962860405,
        inf, 10000.000049346, 98.396094853996, 768.15297955672, inf,
        1.0000000050059, 0.010000983281543, 7.7463475257587, inf, inf,
        137073.10209619, 211719.04367353, inf, inf, 9838.6257712601,
        76811.457712096, inf, inf, 0.00010000000474975, 7.7459668529435, inf,
        inf, 1370730.9694406, 2117190.3656234, inf, inf, 98386.256470787,
        768114.54300982, inf, inf, 9.99984331429e-06, 7.7459254135055, inf,
        inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf,
        inf, inf, inf, inf, inf, inf, inf, inf, inf};
    static const double expected_all_errors[] = {52.063540153765,
        -0.46230400035493, 3.3838561098844, 2.3998800089991, 0, 1.243610958517,
        -15.999200059996, 0, 0, -645.19304347809, 25.058574704815,
        50.290552485276, -29.443199854709, 1.7886888728781, 18.246365467577, 0,
        -1599.8426904211, 0, 561.93151553504, -623.35388309947,
        40.334088238136, -226.15209393773, 0, 0, -1706.6981779472,
        -1316.4685033787, -122.50084953256, -477.61345405917, 0, 0};
    // clang-format on

    EXPECT_VEC_NEAR(expected_all_first_distances, all_first_distances, 1e-5);
    EXPECT_VEC_NEAR(expected_all_errors, all_errors, 1e-2);
}

//---------------------------------------------------------------------------//
/*!
 * Test initialization on or near boundary
 */
class DegenerateBoundaryTest : public Test
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
    real_type eps = -1;
};

void DegenerateBoundaryTest::run(real_type xdir) const
{
    CCylZ cyl(radius);
    CCylZ::Intersections distances = {-1, -1};
    const real_type tol = std::max(1.e-14, 2 * std::fabs(eps));

    // Distance across the cylinder
    const real_type diameter = 2 * radius;

    Real3 pos = {0, 0, 0};
    Real3 dir = {xdir, 0, 0};

    //// Inward boundary ////
    pos[0] = -xdir * (diameter / 2 + eps);
    distances = cyl.calc_intersections(pos, dir, SurfaceState::on);
    EXPECT_SOFT_NEAR(diameter + eps, distances[0], tol);
    EXPECT_EQ(no_intersection(), distances[1]);

    //// Outward boundary ////
    pos[0] = xdir * (diameter / 2 + eps);
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
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
