//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SphereCentered.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SphereCentered.hh"

#include "orange/Constants.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class SphereCenteredTest : public Test
{
  protected:
    using Intersections = SphereCentered::Intersections;

    static constexpr real_type sqrt_third = 1 / constants::sqrt_three;
    static constexpr Real3 inward{-sqrt_third, -sqrt_third, -sqrt_third};
    static constexpr Real3 outward{sqrt_third, sqrt_third, sqrt_third};
};

//---------------------------------------------------------------------------//
TEST_F(SphereCenteredTest, construction)
{
    EXPECT_EQ(SurfaceType::sc, SphereCentered::surface_type());
    EXPECT_EQ(1, SphereCentered::StorageSpan::extent);
    EXPECT_EQ(2, SphereCentered::Intersections{}.size());

    SphereCentered s{3};
    EXPECT_SOFT_EQ(ipow<2>(3), s.radius_sq());

    auto s2 = SphereCentered::from_radius_sq(s.radius_sq());
    EXPECT_SOFT_EQ(s.radius_sq(), s2.radius_sq());
}

TEST_F(SphereCenteredTest, maths)
{
    real_type radius = 4.4;
    SphereCentered s{radius};

    EXPECT_EQ(SignedSense::outside, s.calc_sense({2, 3, 5}));
    EXPECT_EQ(SignedSense::inside, s.calc_sense({2, 3, 1}));

    Real3 const on_surface{
        radius * sqrt_third, radius * sqrt_third, radius * sqrt_third};
    Intersections distances;

    // On surface, inward
    distances = calc_intersections(s, on_surface, inward, SurfaceState::on);
    EXPECT_SOFT_EQ(2 * radius, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // On surface, outward
    distances = calc_intersections(s, on_surface, outward, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // At center
    distances
        = calc_intersections(s, Real3{0, 0, 0}, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(radius, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // Outside, hitting both
    distances = calc_intersections(
        s, Real3{-(radius + 1), 0, 0}, Real3{1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.0, distances[0]);
    EXPECT_SOFT_EQ(1 + 2 * radius, distances[1]);
}

TEST_F(SphereCenteredTest, TEST_IF_CELERITAS_DOUBLE(degenerate))
{
    real_type radius = 4.4;
    SphereCentered s{radius};
    Real3 const on_surface{
        radius * sqrt_third, radius * sqrt_third, radius * sqrt_third};

    // "Not on surface", inward
    auto distances
        = calc_intersections(s, on_surface, inward, SurfaceState::off);
    EXPECT_SOFT_EQ(1e-16, distances[0]);
    EXPECT_SOFT_EQ(2 * radius, distances[1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
