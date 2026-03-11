//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/CylCentered.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/CylCentered.hh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using CCylXTest = Test;
using CCylYTest = Test;
using CCylZTest = Test;

using Intersections = CCylX::Intersections;
using VecReal = std::vector<real_type>;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(CCylXTest, construction)
{
    EXPECT_EQ(1, CCylX::StorageSpan::extent);
    EXPECT_EQ(2, CCylX::Intersections{}.size());

    CCylX c(4.0);
    EXPECT_SOFT_EQ(ipow<2>(4), c.radius_sq());

    real_type const expected_data[] = {ipow<2>(4)};

    EXPECT_VEC_SOFT_EQ(expected_data, c.data());

    auto cy = CCylY::from_radius_sq(c.radius_sq());
    EXPECT_SOFT_EQ(c.radius_sq(), cy.radius_sq());
}

TEST_F(CCylXTest, sense)
{
    CCylX cyl(4.0);

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense(Real3{0, 3, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense(Real3{0, 0, 9}));
}

TEST_F(CCylXTest, normal)
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

TEST_F(CCylXTest, intersect)
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

TEST_F(CCylXTest, intersect_from_surface)
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

TEST_F(CCylYTest, sense)
{
    CCylY cyl(3.0);

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense(Real3{1.5, 0, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense(Real3{3.01, 0, 0}));
}

TEST_F(CCylYTest, intersect)
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

TEST_F(CCylYTest, intersect_from_surface)
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

TEST_F(CCylZTest, sense)
{
    CCylZ cyl(3.0);

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense(Real3{1.5, 0, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense(Real3{3.01, 0, 0}));
}

TEST_F(CCylZTest, calc_intersections)
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

TEST_F(CCylZTest, calc_intersections_on_surface)
{
    CCylZ::Intersections distances;
    real_type const eps = 1.e-4;

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
            Real3{1 - eps, 0, 0}, Real3{1, 0, 0}, SurfaceState::on);
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
TEST_F(CCylZTest, TEST_IF_CELERITAS_DOUBLE(multi_tangent_intersect))
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
            Real3 pos{x, real_type{-10 - 100 * coarse_eps} * v, 0};
            Real3 dir{0, v, 0};

            real_type d;

            // Transport to inside of cylinder
            d = calc_intersections(cyl, pos, dir, SurfaceState::off)[0];
            ASSERT_NE(no_intersection(), d);
            all_first_distances.push_back(d);
            pos[Y] += d * dir[Y];
            all_y.push_back(pos[Y]);

            // Transport to other side of cylinder
            d = calc_intersections(cyl, pos, dir, SurfaceState::on)[0];
            all_distances.push_back(d);
            if (d == no_intersection())
                continue;

            pos[Y] += d * dir[Y];

            // We're done
            d = calc_intersections(cyl, pos, dir, SurfaceState::on)[0];
            EXPECT_EQ(no_intersection(), d);
        }
    }

    constexpr real_type tol{1e-5};
    // clang-format off
    const real_type expected_all_first_distances[] = {9.99869, 9.99869, 0.0001,
        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 9.99869,
        9.99869};
    EXPECT_VEC_NEAR(expected_all_first_distances, all_first_distances, tol);

    const real_type expected_all_y[] = {0.00141421, -0.00141421, 10, -10, 10,
        -10, 10, -10, 10, -10, 0.00141421, -0.00141421};
    EXPECT_VEC_NEAR(expected_all_y, all_y, tol);

    const real_type expected_all_distances[] = {0.00282843, 0.00282843, 20,
        20, 20, 20, 20, 20, 20, 20, 0.00282843, 0.00282843};
    EXPECT_VEC_NEAR(expected_all_distances, all_distances, tol);
    // clang-format on
}

TEST_F(CCylZTest, TEST_IF_CELERITAS_DOUBLE(multi_along_intersect))
{
    CCylZ cyl(30.0);

    VecReal all_first_distances;

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
                dir = make_unit_vector(dir);

                // Transport to inside of cylinder
                real_type d
                    = calc_intersections(cyl, pos, dir, SurfaceState::off)[0];
                all_first_distances.push_back(d);
                if (d == no_intersection())
                    continue;

                // Move the solved distance and make sure we're close to the
                // actual cylinder surface
                axpy(d, dir, &pos);
                real_type boundary_error = std::hypot(pos[0], pos[1])
                                           - real_type(30);

                // Calculate relative to the distance traveled and floating
                // point precision
                real_type rel_error_ulp
                    = boundary_error
                      / (std::numeric_limits<real_type>::epsilon() * d);
                EXPECT_LT(std::fabs(rel_error_ulp), 2000)
                    << "large error " << boundary_error << " ("
                    << rel_error_ulp << ") ULP from " << orig_pos << " along "
                    << dir << " with calculated distance " << d;
            }
        }
    }

    constexpr real_type inf = no_intersection();

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
    // clang-format on

    EXPECT_VEC_NEAR(
        expected_all_first_distances, all_first_distances, real_type{1e-5});
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
    real_type const tol = std::max<real_type>(1.e-14, 2 * std::fabs(eps));

    // Distance across the cylinder
    real_type const diameter = 2 * radius;

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
