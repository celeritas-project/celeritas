//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/ConeAligned.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/ConeAligned.hh"

#include "celeritas_config.h"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(ConeAlignedTest, construction)
{
    ConeX cone{{0, 0, 1}, 0.5};
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1}), cone.origin());
    EXPECT_SOFT_EQ(ipow<2>(0.5), cone.tangent_sq());

    auto coney = ConeY::from_tangent_sq({1, 0, 1}, cone.tangent_sq());
    EXPECT_VEC_SOFT_EQ((Real3{1, 0, 1}), coney.origin());
    EXPECT_SOFT_EQ(ipow<2>(0.5), coney.tangent_sq());
}

TEST(ConeAlignedTest, sense)
{
    ConeX cone{{0, 0, 1}, 0.5};
    EXPECT_EQ(SignedSense::inside, cone.calc_sense({2, 0, 1}));
    EXPECT_EQ(SignedSense::inside, cone.calc_sense({2, 0.99, 1}));
    EXPECT_EQ(SignedSense::outside, cone.calc_sense({2, 1.01, 1}));

    EXPECT_EQ(SignedSense::inside, cone.calc_sense({2, 0, 1}));
    EXPECT_EQ(SignedSense::inside, cone.calc_sense({2, -0.99, 1}));
    EXPECT_EQ(SignedSense::outside, cone.calc_sense({2, -1.01, 1}));

    EXPECT_EQ(SignedSense::inside,
              cone.calc_sense({
                  -4,
                  0,
                  1,
              }));
    EXPECT_EQ(SignedSense::inside,
              cone.calc_sense({
                  -4,
                  0,
                  -0.99,
              }));
    EXPECT_EQ(SignedSense::outside,
              cone.calc_sense({
                  -4,
                  0,
                  -1.01,
              }));
}

TEST(ConeAlignedTest, normal)
{
    ConeX cone{{2, 3, 4}, 0.5};

    EXPECT_VEC_SOFT_EQ(make_unit_vector(Real3{-1, 2, 0}),
                       cone.calc_normal({2 + 2, 3 + 1, 4 + 0}));
}

TEST(ConeAlignedTest, intersection_typical)
{
    // Cone with origin at (1.1, 2.2, 3.3) , slope of +- 2/3
    ConeX cone{{1.1, 2.2, 3.3}, 2. / 3.};

    auto distances = calc_intersections(cone,
                                        {1.1 + 3, 2.2 + 2 + 1.0, 3.3},
                                        {0.0, -1.0, 0.0},
                                        SurfaceState::off);
    EXPECT_SOFT_EQ(1.0, distances[0]);
    EXPECT_SOFT_EQ(5.0, distances[1]);
}

TEST(ConeAlignedTest, intersection_on_surface)
{
    // Cone with origin at (1.1, 2.2, 3.3) , slope of +- 2/3
    ConeX cone{{1.1, 2.2, 3.3}, 2. / 3.};

    auto distances = calc_intersections(
        cone, {1.1 + 3, 2.2 + 2, 3.3}, {0.0, -1.0, 0.0}, SurfaceState::on);
    EXPECT_SOFT_EQ(4.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
}

TEST(ConeAlignedTest, intersection_along_surface)
{
    // Cone with origin at (1.1, 2.2, 3.3) , slope of +- 2/3
    ConeX cone{{1.1, 2.2, 3.3}, 2. / 3.};

    // Along the cone edge heading up and right
    Real3 dir = make_unit_vector(Real3{3.0, 2.0, 0.0});

    // Below lower left sheet
    Real3 pos{1.1 - 3, 2.2 - 2 - 1, 3.3};
    auto distances = calc_intersections(cone, pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(4.5069390943299865, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        // Move to calculated endpoint
        axpy(distances[0] - real_type(1e-10), dir, &pos);
        // Calculate inward direction (near cone, normal is outward dir)
        auto tempdir = cone.calc_normal(pos);
        for (auto& d : tempdir)
        {
            d *= -1;
        }

        auto distances
            = calc_intersections(cone, pos, tempdir, SurfaceState::off);
        EXPECT_SOFT_NEAR(1e-10, distances[0], 0.1);
        EXPECT_SOFT_EQ(2.1633307647466964, distances[1]);
    }

    // Inside left sheet
    distances = calc_intersections(
        cone, {1.1 - 3, 2.2 - 2 + 1, 3.3}, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(2.7041634565979917, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // Inside right sheet
    distances = calc_intersections(
        cone, {1.1 + 3, 2.2 + 2 - 1, 3.3}, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // Outside right sheet
    distances = calc_intersections(
        cone, {1.1 + 3, 2.2 + 2 + 1, 3.3}, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // DOUBLE DEGENERATE: on sheet, traveling along it
    distances = calc_intersections(
        cone, {1.1 + 3, 2.2 + 2, 3.3}, dir, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
}

TEST(ConeAlignedTest, debug_intersect)
{
    ConeX cone{{6.5, 4.0, 0.0}, 2. / 3.};
    Real3 pos{11.209772802620732, 4.7633848351978276, 1.438286089005397};
    Real3 dir{
        -0.97544487122446721, -0.21994072746270818, -0.011549008834416619};

    auto distances = calc_intersections(cone, pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(2.645662863301006, distances[0]);
    EXPECT_SOFT_EQ(9.9221689631064418 - 2.645662863301006, distances[1]);

    axpy(distances[0], dir, &pos);
    real_type expected_next = distances[1] - distances[0];
    distances = calc_intersections(cone, pos, dir, SurfaceState::on);
    EXPECT_SOFT_EQ(expected_next, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    axpy(distances[0], dir, &pos);
    distances = calc_intersections(cone, pos, dir, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
}

TEST(ConeAlignedTest, degenerate_boundary)
{
    Real3 const origin{1.1, 2.2, 3.3};
    real_type const radius = 0.9;
    ConeZ const cone{origin, radius};
    Real3 const dir{1, 0, 0};

    for (auto z : {-1.0, 1.0})
    {
        SCOPED_TRACE(z < 0 ? "below" : "above");
        for (auto eps : {-1e-8, 0.0, 1e-8})
        {
            SCOPED_TRACE(eps < 0 ? "neg" : eps > 0 ? "pos" : "zero");

            real_type const tol = std::max(1.e-14, std::fabs(eps));

            // Distance across the cone at the current point
            real_type const diameter = 2 * std::fabs(z) * radius;

            Real3 pos = origin;
            // Move so that the circular cross section looks like an equivalent
            // cylinder
            pos[2] += 1.0;

            // Left boundary
            pos[0] = origin[0] - diameter / 2 - eps;

            auto distances
                = calc_intersections(cone, pos, dir, SurfaceState::on);
            EXPECT_SOFT_NEAR(diameter + eps, distances[0], tol);
            EXPECT_EQ(no_intersection(), distances[1]);

            // Right boundary
            pos[0] = origin[0] + diameter / 2 + eps;

            distances = calc_intersections(cone, pos, dir, SurfaceState::on);
            EXPECT_EQ(no_intersection(), distances[0]);
            EXPECT_EQ(no_intersection(), distances[1]);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
