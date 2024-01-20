//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/CylAligned.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/CylAligned.hh"

#include "corecel/math/Algorithms.hh"
#include "orange/surf/CylCentered.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using CylTest = Test;
using CylXTest = Test;
using CylYTest = Test;
using CylZTest = Test;
//---------------------------------------------------------------------------//
TEST_F(CylTest, construction)
{
    CylX cyl{{1, 2, 3}, 4};
    EXPECT_SOFT_EQ(2, cyl.origin_u());
    EXPECT_SOFT_EQ(3, cyl.origin_v());
    EXPECT_SOFT_EQ(ipow<2>(4), cyl.radius_sq());
    EXPECT_VEC_EQ((Real3{0, 2, 3}), cyl.calc_origin());

    auto cyly = CylY::from_radius_sq({1, 2, 3}, cyl.radius_sq());
    EXPECT_SOFT_EQ(1, cyly.origin_u());
    EXPECT_SOFT_EQ(3, cyly.origin_v());
    EXPECT_SOFT_EQ(cyl.radius_sq(), cyly.radius_sq());

    CylZ const ccyl{CCylZ{2.5}};
    EXPECT_SOFT_EQ(ipow<2>(2.5), ccyl.radius_sq());
    EXPECT_SOFT_EQ(0, ccyl.origin_u());
    EXPECT_SOFT_EQ(0, ccyl.origin_v());
}

TEST_F(CylXTest, sense)
{
    CylX cyl{{0, 0, 0}, 4.0};

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense({0, 3, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense({0, 0, 9}));
}

TEST_F(CylXTest, normal)
{
    Real3 norm;

    CylX cyl{{0, 1.23, 2.34}, 3.45};

    norm = cyl.calc_normal({1.23, 4.68, 2.34});
    EXPECT_SOFT_EQ(0, norm[0]);
    EXPECT_SOFT_EQ(1, norm[1]);
    EXPECT_SOFT_EQ(0, norm[2]);

    norm = cyl.calc_normal({12345., 2.7728869044748548, 5.42577380894971});
    EXPECT_SOFT_EQ(0, norm[0]);
    EXPECT_SOFT_EQ(0.4472135954999578, norm[1]);
    EXPECT_SOFT_EQ(0.894427190999916, norm[2]);
}

TEST_F(CylXTest, intersect)
{
    // From inside
    CylX cyl{{1234.5, 0, 1}, 3.0};
    auto distances
        = calc_intersections(cyl, {0, 0, 2.5}, {0, 1, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.598076211353316, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting both
    distances[0] = distances[1] = -1;
    distances
        = calc_intersections(cyl, {0, 0, -4.0}, {0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_SOFT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances
        = calc_intersections(cyl, {0, -3, -4.0}, {0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances
        = calc_intersections(cyl, {0, 0, -4.0}, {0, 1, 0}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST_F(CylXTest, intersect_from_surface)
{
    CylX cyl{{0, 1.23, 2.34}, 3.45};

    // One intercept

    auto distances = calc_intersections(
        cyl, {1.23, 4.68, 2.34}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(6.9, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances
        = calc_intersections(cyl,
                             {12345., 2.7728869044748548, 5.42577380894971},
                             {0.6, 0, -0.8},
                             SurfaceState::on);
    EXPECT_SOFT_EQ(7.714434522374273, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // No intercepts

    distances = calc_intersections(
        cyl, {1.23, 4.68, 2.34}, {0, 1, 0}, SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances
        = calc_intersections(cyl,
                             {12345., 2.7728869044748548, 5.42577380894971},
                             {0.6, 0, 0.8},
                             SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

//---------------------------------------------------------------------------//

TEST_F(CylYTest, sense)
{
    CylY cyl{{1, 1234.5, 0}, 3.0};

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense({2.5, 0, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense({4.01, 0, 0}));
}

TEST_F(CylYTest, normal)
{
    Real3 norm;

    CylY cyl{{1.23, 0, 2.34}, 3.45};

    norm = cyl.calc_normal({4.68, 1.23, 2.34});
    EXPECT_SOFT_EQ(1, norm[0]);
    EXPECT_SOFT_EQ(0, norm[1]);
    EXPECT_SOFT_EQ(0, norm[2]);

    norm = cyl.calc_normal({2.7728869044748548, 12345., 5.42577380894971});
    EXPECT_SOFT_EQ(0.4472135954999578, norm[0]);
    EXPECT_SOFT_EQ(0, norm[1]);
    EXPECT_SOFT_EQ(0.894427190999916, norm[2]);
}

TEST_F(CylYTest, intersect)
{
    // From inside
    CylY cyl{{1, 1234.5, 0}, 3.0};
    auto distances
        = calc_intersections(cyl, {2.5, 0, 0}, {0, 0, 1}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.598076211353316, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting both
    distances[0] = distances[1] = -1;
    distances
        = calc_intersections(cyl, {-4.0, 0, 0}, {1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    EXPECT_SOFT_EQ(8.0, distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances
        = calc_intersections(cyl, {-4.0, 0, -3}, {1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances
        = calc_intersections(cyl, {-4.0, 0, 0}, {0, 0, 1}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST_F(CylYTest, intersect_from_surface)
{
    CylY cyl{{1.23, 0, 2.34}, 3.45};

    // One intercept

    auto distances = calc_intersections(
        cyl, {4.68, 1.23, 2.34}, {-1, 0, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(6.9, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances
        = calc_intersections(cyl,
                             {2.7728869044748548, 12345., 5.42577380894971},
                             {0, 0.6, -0.8},
                             SurfaceState::on);
    EXPECT_SOFT_EQ(7.714434522374273, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // No intercepts

    distances = calc_intersections(
        cyl, {4.68, 1.23, 2.34}, {1, 0, 0}, SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances
        = calc_intersections(cyl,
                             {2.7728869044748548, 12345., 5.42577380894971},
                             {0, 0.6, 0.8},
                             SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

//---------------------------------------------------------------------------//

TEST_F(CylZTest, sense)
{
    CylZ cyl{{1, 0, 1234.5}, 3.0};

    EXPECT_EQ(SignedSense::inside, cyl.calc_sense({2.5, 0, 0}));
    EXPECT_EQ(SignedSense::outside, cyl.calc_sense({4.01, 0, 0}));
}

TEST_F(CylZTest, normal)
{
    Real3 norm;

    CylZ cyl{{1.23, 2.34, 0}, 3.45};

    norm = cyl.calc_normal({4.68, 2.34, 1.23});
    EXPECT_SOFT_EQ(1, norm[0]);
    EXPECT_SOFT_EQ(0, norm[1]);
    EXPECT_SOFT_EQ(0, norm[2]);

    norm = cyl.calc_normal({2.7728869044748548, 5.42577380894971, 12345.});
    EXPECT_SOFT_EQ(0.4472135954999578, norm[0]);
    EXPECT_SOFT_EQ(0.894427190999916, norm[1]);
    EXPECT_SOFT_EQ(0, norm[2]);
}

TEST_F(CylZTest, calc_intersections)
{
    // From inside
    CylZ cyl{{1, 0, 1234.5}, 3.0};
    auto distances
        = calc_intersections(cyl, {2.5, 0, 0}, {0, 1, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.598076211353316, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting both
    distances[0] = distances[1] = -1;
    distances
        = calc_intersections(cyl, {-4.0, 0, 0}, {1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(2.0, distances[0]);
    // TODO: we should calculate intersection to both surfaces
    // EXPECT_EQ(8.0,      distances[1]);

    // From outside, tangent
    distances[0] = distances[1] = -1;
    distances
        = calc_intersections(cyl, {-4.0, -3, 0}, {1, 0, 0}, SurfaceState::off);

    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // From outside, hitting neither
    distances[0] = distances[1] = -1;
    distances
        = calc_intersections(cyl, {-4.0, 0, 0}, {0, 1, 0}, SurfaceState::off);

    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST_F(CylZTest, calc_intersections_on_surface)
{
    CylZ cyl{{1.23, 2.34, 0}, 3.45};

    // One intercept

    auto distances = calc_intersections(
        cyl, {4.68, 2.34, 1.23}, {-1, 0, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(6.9, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances
        = calc_intersections(cyl,
                             {2.7728869044748548, 5.42577380894971, 12345.},
                             {0, -0.8, 0.6},
                             SurfaceState::on);
    EXPECT_SOFT_EQ(7.714434522374273, distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    // No intercepts

    distances = calc_intersections(
        cyl, {4.68, 2.34, 1.23}, {1, 0, 0}, SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);

    distances
        = calc_intersections(cyl,
                             {2.7728869044748548, 5.42577380894971, 12345.},
                             {0, 0.8, 0.6},
                             SurfaceState::on);
    EXPECT_EQ(no_intersection(), distances[0]);
    EXPECT_EQ(no_intersection(), distances[1]);
}

TEST_F(CylZTest, calc_intersections_degenerate)
{
    real_type const eps = std::sqrt(coarse_eps);

    {
        CylZ cyl{{1.0, 0.0, 0}, 1.0};

        // Heading toward, slightly inside
        auto distances = calc_intersections(
            cyl, {eps, 0, 0}, {1, 0, 0}, SurfaceState::off);
        EXPECT_SOFT_EQ(2 - eps, distances[0]);
        EXPECT_SOFT_EQ(no_intersection(), distances[1]);

        // Heading away, slightly inside
        distances = calc_intersections(
            cyl, {eps, 0, 0}, {-1, 0, 0}, SurfaceState::off);
        EXPECT_SOFT_EQ(eps, distances[0]);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Tangent, inside, off surface
        distances = calc_intersections(
            cyl, {eps, 0, 0}, {0, 1, 0}, SurfaceState::off);
        EXPECT_SOFT_NEAR(std::sqrt(2 * eps), distances[0], eps);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Tangent, inside, on surface
        distances = calc_intersections(
            cyl, {eps, 0, 0}, {0, 1, 0}, SurfaceState::on);
        EXPECT_EQ(no_intersection(), distances[1]);
        EXPECT_EQ(no_intersection(), distances[0]);

        // Heading in
        distances = calc_intersections(
            cyl, {eps, 0, 0}, {1, 0, 0}, SurfaceState::off);
        EXPECT_SOFT_NEAR(2 + eps, distances[0], eps);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Heading away
        distances = calc_intersections(
            cyl, {2 - eps, 0, 0}, {1, 0, 0}, SurfaceState::off);
        EXPECT_SOFT_NEAR(eps, distances[0], eps);
        EXPECT_EQ(no_intersection(), distances[1]);

        // Tangent
        distances = calc_intersections(
            cyl, {eps, 0, 0}, {0, 1, 0}, SurfaceState::off);
        EXPECT_SOFT_EQ(std::sqrt(2 * eps - eps * eps), distances[0]);
        EXPECT_EQ(no_intersection(), distances[1]);
    }
    {
        CylZ cyl{{1.23, 2.34, 0}, 3.45};

        auto distances = calc_intersections(cyl,
                                            {real_type{4.68} - eps, 2.34, 1.23},
                                            {-1, 0, 0},
                                            SurfaceState::off);
        EXPECT_SOFT_EQ(6.9 - eps, distances[0]);
        EXPECT_EQ(no_intersection(), distances[1]);

        distances = calc_intersections(cyl,
                                       {real_type{4.68} - eps, 2.34, 1.23},
                                       {1, 0, 0},
                                       SurfaceState::off);
        EXPECT_SOFT_NEAR(eps, distances[0], eps);
        EXPECT_EQ(no_intersection(), distances[1]);
    }
}

//---------------------------------------------------------------------------//

TEST_F(CylZTest, TEST_IF_CELERITAS_DOUBLE(degenerate_boundary))
{
#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
    constexpr double coarse_eps{1e-8};
#elif CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
    constexpr float coarse_eps{1e-4};
#endif
    Real3 const origin{1.1, 2.2, 3.3};
    for (real_type radius : {0.9, 1.0})
    {
        CylZ const cyl{origin, radius};
        for (real_type xdir : {-1.0, 1.0})
        {
            SCOPED_TRACE(xdir < 0 ? "leftward" : "rightward");
            for (real_type eps : {-coarse_eps, real_type{0}, coarse_eps})
            {
                SCOPED_TRACE(eps < 0 ? "neg" : eps > 0 ? "pos" : "zero");

                real_type const tol
                    = std::max<real_type>(1.e-14, 2 * std::fabs(eps));

                // Distance across the cylinder
                real_type const diameter = 2 * radius;

                Real3 pos = origin;
                Real3 dir = {xdir, 0, 0};

                // >>> Inward boundary
                pos[0] = origin[0] - xdir * (diameter / 2 + eps);
                auto distances
                    = calc_intersections(cyl, pos, dir, SurfaceState::on);
                EXPECT_SOFT_NEAR(diameter + eps, distances[0], tol);
                EXPECT_EQ(no_intersection(), distances[1]);

                // >>> Outward boundary
                pos[0] = origin[0] + xdir * (diameter / 2 + eps);
                distances = calc_intersections(cyl, pos, dir, SurfaceState::on);
                EXPECT_EQ(no_intersection(), distances[0]);
                EXPECT_EQ(no_intersection(), distances[1]);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
