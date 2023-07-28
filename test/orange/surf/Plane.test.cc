//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Plane.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Plane.hh"

#include "orange/Constants.hh"

#include "celeritas_test.hh"

using celeritas::constants::sqrt_two;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PlaneTest : public Test
{
  protected:
    using Intersections = Plane::Intersections;

    real_type calc_intersection(Plane const& surf,
                                Real3 pos,
                                Real3 dir,
                                SurfaceState s = SurfaceState::off)
    {
        static_assert(sizeof(Plane::Intersections) == sizeof(real_type),
                      "Expected plane to have a single intercept");
        return surf.calc_intersections(pos, dir, s)[0];
    }
};

TEST_F(PlaneTest, tracking)
{
    // Make a rotated plane in the xy axis
    Plane p{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two};

    // Get a point that should have positive sense
    Real3 x{{5.41421356, 1.41421356, 0.0}};
    EXPECT_EQ(SignedSense::outside, p.calc_sense(x));

    // Calc intersections
    Real3 dir = {-0.70710678, -0.70710678, 0.0};
    normalize_direction(&dir);
    EXPECT_SOFT_NEAR(
        2.0, calc_intersection(p, x, dir, SurfaceState::off), 1.e-6);

    // Pick a direction such that n\cdot\Omega > 0
    dir = {1.0, 2.0, 3.0};
    normalize_direction(&dir);
    EXPECT_EQ(no_intersection(),
              calc_intersection(p, x, dir, SurfaceState::off));

    // Pick a direction that hits the plane
    dir = {-1, 0.1, 3.0};
    normalize_direction(&dir);
    EXPECT_SOFT_NEAR(9.9430476983098171,
                     calc_intersection(p, x, dir, SurfaceState::off),
                     1.e-6);

    // Place a point on the negative sense
    x = {1.87867966, -2.12132034, 0.0};
    EXPECT_EQ(SignedSense::inside, p.calc_sense(x));

    // Pick a direction such that n\cdot\Omega < 0
    dir = {-1.0, -2.0, 3.0};
    normalize_direction(&dir);
    EXPECT_EQ(no_intersection(),
              calc_intersection(p, x, dir, SurfaceState::off));

    // Pick a direction that hits the plane
    dir = {1, 0.1, 3.0};
    normalize_direction(&dir);
    EXPECT_SOFT_NEAR(12.202831266107504,
                     calc_intersection(p, x, dir, SurfaceState::off),
                     1.e-6);

    // Place a point on the surface
    x = {2.0, 2.0, 0.0};
    EXPECT_EQ(no_intersection(),
              calc_intersection(p, x, dir, SurfaceState::on));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
