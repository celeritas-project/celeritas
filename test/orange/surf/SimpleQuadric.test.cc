//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SimpleQuadric.hh"

#include "corecel/math/Algorithms.hh"
#include "orange/Constants.hh"
#include "orange/surf/ConeAligned.hh"
#include "orange/surf/CylAligned.hh"
#include "orange/surf/Plane.hh"
#include "orange/surf/Sphere.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(SimpleQuadricTest, construction)
{
    using constants::sqrt_two;
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
