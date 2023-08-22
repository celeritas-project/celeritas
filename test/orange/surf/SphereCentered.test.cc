//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SphereCentered.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SphereCentered.hh"

#include "orange/Constants.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using Intersections = SphereCentered::Intersections;

constexpr real_type sqrt_third = 1 / constants::sqrt_three;

//---------------------------------------------------------------------------//
TEST(SphereCenteredTest, construction)
{
    EXPECT_EQ(SurfaceType::sc, SphereCentered::surface_type());
    EXPECT_EQ(1, SphereCentered::Storage::extent);
    EXPECT_EQ(2, SphereCentered::Intersections{}.size());

    SphereCentered s{3};
    EXPECT_SOFT_EQ(ipow<2>(3), s.radius_sq());

    auto s2 = SphereCentered::from_radius_sq(s.radius_sq());
    EXPECT_SOFT_EQ(s.radius_sq(), s2.radius_sq());
}

TEST(SphereCenteredTest, maths)
{
    real_type radius = 4.4;
    SphereCentered s{radius};

    EXPECT_EQ(SignedSense::outside, s.calc_sense({2, 3, 5}));
    EXPECT_EQ(SignedSense::inside, s.calc_sense({2, 3, 1}));

    const Real3 on_surface{
        radius * sqrt_third, radius * sqrt_third, radius * sqrt_third};
    const Real3 inward{-sqrt_third, -sqrt_third, -sqrt_third};
    const Real3 outward{sqrt_third, sqrt_third, sqrt_third};

    Intersections distances;

    // On surface, inward
    distances = s.calc_intersections(on_surface, inward, SurfaceState::on);
    EXPECT_SOFT_EQ(2 * radius, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // "Not on surface", inward
    distances = s.calc_intersections(on_surface, inward, SurfaceState::off);
    EXPECT_SOFT_EQ(1e-16, distances[0]);
    EXPECT_SOFT_EQ(2 * radius, distances[1]);

    // On surface, outward
    distances = s.calc_intersections(on_surface, outward, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // At center
    distances
        = s.calc_intersections(Real3{0, 0, 0}, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(radius, distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);

    // Outside, hitting both
    distances = s.calc_intersections(
        Real3{-(radius + 1), 0, 0}, Real3{1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.0, distances[0]);
    EXPECT_SOFT_EQ(1 + 2 * radius, distances[1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
