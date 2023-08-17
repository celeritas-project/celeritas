//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SimpleQuadric.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SimpleQuadric.hh"

#include "corecel/math/Algorithms.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(Ellipsoid, origin)
{
    // 1 x 2.5 x .3 radii
    const Real3 second{ipow<2>(2.5) * ipow<2>(0.3),
                       ipow<2>(1.0) * ipow<2>(0.3),
                       ipow<2>(1.0) * ipow<2>(2.5)};
    const Real3 first{0, 0, 0};
    const real_type zeroth = -1 * ipow<2>(2.5) * ipow<2>(0.3);
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

TEST(Ellipsoid, translated)
{
    // Radius: {1, 2.5, .3} centered at {2, 3, 4}
    SimpleQuadric sq{SimpleQuadric{{0.5625, 0.09, 6.25}, {0, 0, 0}, -0.5625},
                     {2, 3, 4}};

    auto distances
        = sq.calc_intersections({-0.5, 3, 4}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.5, distances[0]);
    EXPECT_SOFT_EQ(1.5 + 2.0, distances[1]);
    distances
        = sq.calc_intersections({2, 5.5, 4}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
    distances = sq.calc_intersections({2, 3, 4}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_EQ(0.3, distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
