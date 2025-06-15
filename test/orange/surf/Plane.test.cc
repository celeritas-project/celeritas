//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Plane.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Plane.hh"

#include "corecel/Constants.hh"
#include "orange/surf/PlaneAligned.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
constexpr auto sqrt_two = real_type{constants::sqrt_two};

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

TEST_F(PlaneTest, construction)
{
    // Make a rotated plane in the xy axis
    Plane p{{1 / sqrt_two, 1 / sqrt_two, 0.0}, {2 / sqrt_two, 2 / sqrt_two, 2}};
    EXPECT_VEC_SOFT_EQ((Real3{1 / sqrt_two, 1 / sqrt_two, 0}), p.normal());
    EXPECT_SOFT_EQ(2, p.displacement());

    Plane px{PlaneX{1.25}};
    EXPECT_VEC_SOFT_EQ((Real3{1, 0, 0}), px.normal());
    EXPECT_SOFT_EQ(1.25, px.displacement());

    Plane py{PlaneY{2.25}};
    EXPECT_VEC_SOFT_EQ((Real3{0, 1, 0}), py.normal());

    // Construct from three points, in this case a plane passing through the
    // point (1, 2, 3) with slope (1, 1, 1). Specifying the points in clockwise
    // order gives a negative normal.
    Plane p2{Real3{2, 1, 3}, Real3{-3, 5, 4}, Real3{4, 7, -5}};
    EXPECT_SOFT_EQ(-2 * sqrt(3), p2.displacement());
    real_type dir = sqrt(3) / 3;
    EXPECT_VEC_SOFT_EQ(Real3({-dir, -dir, -dir}), p2.normal());

    // Specifying the points in counterclockwise order flips the displacement
    // and normal
    Plane p3{Real3{2, 1, 3}, Real3{4, 7, -5}, Real3{-3, 5, 4}};
    EXPECT_SOFT_EQ(2 * sqrt(3), p3.displacement());
    EXPECT_VEC_SOFT_EQ(Real3({dir, dir, dir}), p3.normal());
}

TEST_F(PlaneTest, tracking)
{
    // Make a rotated plane in the xy axis
    Plane p{{1 / sqrt_two, 1 / sqrt_two, 0.0}, 2 * sqrt_two};

    // Get a point that should have positive sense
    Real3 x{{5.41421356, 1.41421356, 0.0}};
    EXPECT_EQ(SignedSense::outside, p.calc_sense(x));

    // Calc intersections
    Real3 dir = make_unit_vector(Real3{-0.70710678, -0.70710678, 0.0});
    EXPECT_SOFT_NEAR(
        2.0, calc_intersection(p, x, dir, SurfaceState::off), 1.e-6);

    // Pick a direction such that n\cdot\Omega > 0
    dir = make_unit_vector(Real3{1.0, 2.0, 3.0});
    EXPECT_EQ(no_intersection(),
              calc_intersection(p, x, dir, SurfaceState::off));

    // Pick a direction that hits the plane
    dir = make_unit_vector(Real3{-1, 0.1, 3.0});
    EXPECT_SOFT_NEAR(9.9430476983098171,
                     calc_intersection(p, x, dir, SurfaceState::off),
                     1.e-6);

    // Place a point on the negative sense
    x = {1.87867966, -2.12132034, 0.0};
    EXPECT_EQ(SignedSense::inside, p.calc_sense(x));

    // Pick a direction such that n\cdot\Omega < 0
    dir = make_unit_vector(Real3{-1.0, -2.0, 3.0});
    EXPECT_EQ(no_intersection(),
              calc_intersection(p, x, dir, SurfaceState::off));

    // Pick a direction that hits the plane
    dir = make_unit_vector(Real3{1, 0.1, 3.0});
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
