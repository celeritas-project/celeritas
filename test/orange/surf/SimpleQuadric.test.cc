//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SimpleQuadric.hh"

#include "corecel/Constants.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "orange/surf/ConeAligned.hh"
#include "orange/surf/CylAligned.hh"
#include "orange/surf/Plane.hh"
#include "orange/surf/Sphere.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
constexpr auto sqrt_two = real_type{constants::sqrt_two};

TEST(SimpleQuadricTest, construction)
{
    // Plane
    SimpleQuadric p{Plane{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two}};

    auto distances
        = p.calc_intersections({0, 0, 0}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(4.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // Sphere
    SimpleQuadric sph{Sphere{{1, 2, 3}, 0.5}};

    distances = sph.calc_intersections({1, 2, 2}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_EQ(0.5, distances[0]);
    EXPECT_SOFT_EQ(1.5, distances[1]);

    // Cone along x axis
    SimpleQuadric cx{ConeX{{1.1, 2.2, 3.3}, 2. / 3.}};

    distances = cx.calc_intersections(
        {1.1 + 3, 2.2 + 2 + 1.0, 3.3}, {0.0, -1.0, 0.0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.0, distances[0]);
    EXPECT_SOFT_EQ(5.0, distances[1]);

    // Cylinder with radius 2 centered at {2, 3, 0}
    SimpleQuadric cz{CylZ{{2, 3, 0}, 2}};

    distances
        = cz.calc_intersections({-0.5, 3, 0}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(0.5, distances[0]);
    EXPECT_SOFT_EQ(0.5 + 4.0, distances[1]);

    EXPECT_VEC_SOFT_EQ((Real3{0, 1, 0}), cz.calc_normal({2, 5, 0}));
    EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), cz.calc_normal({0, 3, 0}));
}

TEST(SimpleQuadricTest, ellipsoid)
{
    // 1 x 2.5 x .3 radii
    Real3 const second{ipow<2>(2.5) * ipow<2>(0.3),
                       ipow<2>(1.0) * ipow<2>(0.3),
                       ipow<2>(1.0) * ipow<2>(2.5)};
    real_type const zeroth = -ipow<2>(1) * ipow<2>(2.5) * ipow<2>(0.3);

    SimpleQuadric sq{second, {0, 0, 0}, zeroth};

    EXPECT_VEC_SOFT_EQ(second, sq.second());
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), sq.first());
    EXPECT_SOFT_EQ(zeroth, sq.zeroth());

    // Test intersections along major axes
    EXPECT_EQ(SignedSense::outside, sq.calc_sense({-2.5, 0, 0}));
    auto distances
        = sq.calc_intersections({-2.5, 0, 0}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.5, distances[0]);
    EXPECT_SOFT_EQ(1.5 + 2.0, distances[1]);

    EXPECT_EQ(SignedSense::on, sq.calc_sense({0, 2.5, 0}));
    distances
        = sq.calc_intersections({0, 2.5, 0}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    EXPECT_EQ(SignedSense::inside, sq.calc_sense({0, 0, 0}));
    distances = sq.calc_intersections({0, 0, 0}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_EQ(0.3, distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);

    // Test normals
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, -1}), sq.calc_normal({0, 0, -0.3}));
    EXPECT_VEC_SOFT_EQ((Real3{0, 1, 0}), sq.calc_normal({0, 2.5, 0}));
    EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), sq.calc_normal({-1, 0, 0}));
}

// This ellipsoid is tiny but scaled incorrectly so that the quadratic solver
// sees a loss of precision
TEST(SimpleQuadricTest, scaled_ellipsoid)
{
    Real3 const radii{1e-3, 2.5e-3, 0.3e-3};
    Real3 const second{ipow<2>(radii[1]) * ipow<2>(radii[2]),
                       ipow<2>(radii[2]) * ipow<2>(radii[0]),
                       ipow<2>(radii[0]) * ipow<2>(radii[1])};
    real_type const zeroth = -ipow<2>(radii[0]) * ipow<2>(radii[1])
                             * ipow<2>(radii[2]);
    SimpleQuadric sq{second, {0, 0, 0}, zeroth};

    // Test intersections along major axes: should be like above (but scaled by
    // 1e-3), but all intersections are missed due to the 'a' term being too
    // small.
    EXPECT_EQ(SignedSense::outside, sq.calc_sense({-2.5e-3, 0, 0}));
    auto distances
        = sq.calc_intersections({-2.5e-3, 0, 0}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    distances
        = sq.calc_intersections({0, 2.5e-3, 0}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    EXPECT_EQ(SignedSense::inside, sq.calc_sense({0, 0, 0}));
    distances = sq.calc_intersections({0, 0, 0}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
}

TEST(SimpleQuadricTest, small_ellipsoid)
{
    // 1 x 2.5 x .3 radii
    Real3 const second{ipow<2>(2.5) * ipow<2>(0.3),
                       ipow<2>(1.0) * ipow<2>(0.3),
                       ipow<2>(1.0) * ipow<2>(2.5)};
    Real3 const first{0, 0, 0};
    real_type const zeroth = -1 * ipow<2>(2.5) * ipow<2>(0.3);
    SimpleQuadric sq{second, first, zeroth};

    EXPECT_VEC_SOFT_EQ(second, sq.second());
    EXPECT_VEC_SOFT_EQ(first, sq.first());
    EXPECT_SOFT_EQ(zeroth, sq.zeroth());

    // Test intersections along major axes
    auto distances
        = sq.calc_intersections({-2.5, 0, 0}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.5, distances[0]);
    EXPECT_SOFT_EQ(1.5 + 2.0, distances[1]);
    distances
        = sq.calc_intersections({0, 2.5, 0}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
    distances = sq.calc_intersections({0, 0, 0}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_EQ(0.3, distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);

    // Test normals
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, -1}), sq.calc_normal({0, 0, -0.3}));
    EXPECT_VEC_SOFT_EQ((Real3{0, 1, 0}), sq.calc_normal({0, 2.5, 0}));
    EXPECT_VEC_SOFT_EQ((Real3{-1, 0, 0}), sq.calc_normal({-1, 0, 0}));
}

TEST(SimpleQuadricTest, safety)
{
    {
        SCOPED_TRACE("sphere");

        SimpleQuadric const sq{Sphere{{1, 2, 3}, 2}};

        // Exact inside: radius minus distance from center
        EXPECT_SOFT_EQ(2, sq.calc_safety({1, 2, 3}));
        EXPECT_SOFT_EQ(1, sq.calc_safety({1, 2, 4}));

        // Outside: f/|grad f| = (25 - 4) / 10, less than the true 3
        EXPECT_SOFT_EQ(2.1, sq.calc_safety({1, 2, 8}));
        EXPECT_LE(sq.calc_safety({1, 2, 8}), 3);
    }
    {
        SCOPED_TRACE("ellipsoid");

        // 1 x 2.5 x .3 radii
        SimpleQuadric const sq{{ipow<2>(2.5) * ipow<2>(0.3),
                                ipow<2>(1.0) * ipow<2>(0.3),
                                ipow<2>(1.0) * ipow<2>(2.5)},
                               {0, 0, 0},
                               -ipow<2>(1) * ipow<2>(2.5) * ipow<2>(0.3)};

        EXPECT_SOFT_EQ(0, sq.calc_safety({0, 2.5, 0}));

        // Exact at the center: the smallest semi-axis
        EXPECT_SOFT_EQ(0.3, sq.calc_safety({0, 0, 0}));

        // Nearly exact just outside and inside the surface
        EXPECT_SOFT_NEAR(
            1e-4, sq.calc_safety({0, 0, 0.3 + 1e-4}), real_type(1e-3));
        EXPECT_SOFT_NEAR(
            1e-4, sq.calc_safety({0, 0, 0.3 - 1e-4}), real_type(1e-3));

        // Outside on the x axis: f/|grad f| = 1.6875 / 2.25, less than the
        // true distance 1 to the vertex (every surface point has x <= 1)
        EXPECT_SOFT_EQ(0.75, sq.calc_safety({2, 0, 0}));
        EXPECT_LE(sq.calc_safety({2, 0, 0}), 1);

        // No closed-form nearest point: compare to a sampled distance
        std::vector<real_type> safeties;
        std::vector<real_type> sampled_distances;

        for (Real3 const& pos :
             {Real3{0.5, 1, 0.1}, Real3{-1, 3, 1}, Real3{0.1, 0.1, 0.25}})
        {
            auto safety = sq.calc_safety(pos);
            auto sampled_distance = calc_sampled_distance(sq, pos);

            safeties.push_back(safety);
            sampled_distances.push_back(sampled_distance);

            EXPECT_LE(safety, sampled_distance) << "at " << repr(pos);
        }
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            auto expected_safeties
                = {0.1246283146222, 0.56200639784968, 0.048228066092882};
            auto expected_sampled_distances
                = {0.12758949034185, 1.3449967588974, 0.048235728171243};
            EXPECT_VEC_SOFT_EQ(expected_safeties, safeties);
            EXPECT_VEC_SOFT_EQ(expected_sampled_distances, sampled_distances);
        }
    }
    {
        SCOPED_TRACE("hyperboloid");

        // One sheet: x^2 + y^2 - z^2 = 1
        SimpleQuadric const sq{{1, 1, -1}, {0, 0, 0}, -1};

        // Exact at the waist center
        EXPECT_SOFT_EQ(1, sq.calc_safety({0, 0, 0}));

        // Above the waist on the axis: the nearest points are at z = 2.5,
        // minimizing 1 + z^2 + (z - 5)^2
        EXPECT_SOFT_EQ(2.14142842854285, sq.calc_safety({0, 0, 5}));
        EXPECT_LE(sq.calc_safety({0, 0, 5}), std::sqrt(real_type(13.5)));

        // Outside the waist: the nearest points are at rho = 1.5,
        // minimizing (rho - 3)^2 + rho^2 - 1
        EXPECT_SOFT_EQ(1.1231056256176606, sq.calc_safety({3, 0, 0}));
        EXPECT_LE(sq.calc_safety({3, 0, 0}), std::sqrt(real_type(3.5)));

        // No closed-form nearest point: compare to a sampled distance
        std::vector<real_type> safeties;
        std::vector<real_type> sampled_distances;

        for (Real3 const& pos : {Real3{1, 1, 2}, Real3{0.5, 0.2, 0.3}})
        {
            auto safety = sq.calc_safety(pos);
            auto sampled_distance = calc_sampled_distance(sq, pos);

            safeties.push_back(safety);
            sampled_distances.push_back(sampled_distance);

            EXPECT_LE(safety, sampled_distance) << "at " << repr(pos);
        }
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            auto expected_safeties = {0.55051025721682, 0.46983664882312};
            auto expected_sampled_distances
                = {0.61914647005653, 0.49158749109334};
            EXPECT_VEC_SOFT_EQ(expected_safeties, safeties);
            EXPECT_VEC_SOFT_EQ(expected_sampled_distances, sampled_distances);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
