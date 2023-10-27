//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Sphere.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/Sphere.hh"

#include "corecel/math/Algorithms.hh"
#include "orange/Constants.hh"
#include "orange/surf/SphereCentered.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class SphereTest : public Test
{
  protected:
    using Intersections = Sphere::Intersections;

    static constexpr real_type sqrt_third = 1 / constants::sqrt_three;
    static constexpr Real3 inward{-sqrt_third, -sqrt_third, -sqrt_third};
    static constexpr Real3 outward{sqrt_third, sqrt_third, sqrt_third};
};

//---------------------------------------------------------------------------//
TEST_F(SphereTest, construction)
{
    Sphere s{{-1.1, 2.2, -3.3}, 4.4};
    EXPECT_VEC_SOFT_EQ((Real3{-1.1, 2.2, -3.3}), s.origin());
    EXPECT_SOFT_EQ(ipow<2>(4.4), s.radius_sq());

    auto s2 = Sphere::from_radius_sq({1, 2, 3}, s.radius_sq());
    EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), s2.origin());
    EXPECT_SOFT_EQ(s.radius_sq(), s2.radius_sq());

    Sphere sc{SphereCentered{2.5}};
    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), sc.origin());
    EXPECT_SOFT_EQ(ipow<2>(2.5), sc.radius_sq());
}

//---------------------------------------------------------------------------//
TEST_F(SphereTest, basic)
{
    EXPECT_EQ(SurfaceType::s, Sphere::surface_type());
    EXPECT_EQ(4, Sphere::Storage::extent);
    EXPECT_EQ(2, Sphere::Intersections{}.size());

    const Real3 origin{-1.1, 2.2, -3.3};
    real_type radius = 4.4;

    Sphere s{origin, radius};
    EXPECT_VEC_SOFT_EQ(origin, s.origin());
    EXPECT_SOFT_EQ(ipow<2>(radius), s.radius_sq());

    EXPECT_EQ(SignedSense::outside, s.calc_sense({4, 5, 5}));
    EXPECT_EQ(SignedSense::inside, s.calc_sense({1, 2, -3}));

    Real3 const on_surface{origin[0] + radius * sqrt_third,
                           origin[1] + radius * sqrt_third,
                           origin[2] + radius * sqrt_third};

    Intersections distances;

    // On surface, inward
    distances = calc_intersections(s, on_surface, inward, SurfaceState::on);
    EXPECT_SOFT_EQ(2 * radius, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // On surface, outward
    distances = calc_intersections(s, on_surface, outward, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
}

//---------------------------------------------------------------------------//
TEST_F(SphereTest, TEST_IF_CELERITAS_DOUBLE(degenerate))
{
    EXPECT_EQ(SurfaceType::s, Sphere::surface_type());
    EXPECT_EQ(4, Sphere::Storage::extent);
    EXPECT_EQ(2, Sphere::Intersections{}.size());

    Real3 const origin{-1.1, 2.2, -3.3};
    real_type radius = 4.4;

    Sphere s{origin, radius};
    Real3 const on_surface{origin[0] + radius * sqrt_third,
                           origin[1] + radius * sqrt_third,
                           origin[2] + radius * sqrt_third};

    Intersections distances;

    // "Not on surface", inward
    distances = calc_intersections(s, on_surface, inward, SurfaceState::off);

    EXPECT_SOFT_EQ(1e-16, distances[0]);
    EXPECT_SOFT_EQ(2 * radius, distances[1]);
    // At center
    distances = calc_intersections(s, origin, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(radius, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // Outside, hitting both
    distances = calc_intersections(
        s, Real3{-6.5, 2.2, -3.3}, Real3{1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.0, distances[0]);
    EXPECT_SOFT_EQ(1 + 2 * radius, distances[1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
