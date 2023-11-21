//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SurfaceFunctors.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/SurfaceFunctors.hh"

#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SurfaceTypeTraits.hh"
#include "orange/surf/detail/AllSurfaces.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SurfaceFunctorsTest : public ::celeritas::test::Test
{
  protected:
    PlaneX px_{1.25};
    Sphere s_{{2.25, 1, 0}, 1.25};
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_sense)
{
    Real3 pos{0.9, 0, 0};
    CalcSense calc{pos};

    EXPECT_EQ(SignedSense::inside, calc(px_));
    EXPECT_EQ(SignedSense::outside, calc(s_));

    pos = {1.0, 0, 0};
    EXPECT_EQ(SignedSense::inside, calc(px_));
    EXPECT_EQ(SignedSense::outside, calc(s_));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, num_intersections)
{
    EXPECT_EQ(1, visit_surface_type(NumIntersections{}, SurfaceType::px));
    EXPECT_EQ(2, visit_surface_type(NumIntersections{}, SurfaceType::sc));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_normal)
{
    Real3 pos;
    CalcNormal calc_normal{pos};

    pos = {1.25, 1, 1};
    EXPECT_EQ(Real3({1, 0, 0}), calc_normal(px_));
    pos = {2.25, 2.25, 0};
    EXPECT_EQ(Real3({0, 1, 0}), calc_normal(s_));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_safety_distance)
{
    Real3 pos;
    CalcSafetyDistance calc_distance{pos};

    real_type eps = 1e-4;
    pos = {real_type{1.25} + eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(s_));

    pos = {real_type{1.25}, 1, 0};
    EXPECT_SOFT_EQ(0, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25, calc_distance(s_));

    pos = {real_type{1.25} - eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25 - eps, calc_distance(s_));

    pos = {real_type{1} - eps, 1, 0};
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(px_));
    EXPECT_SOFT_EQ(eps, calc_distance(s_));

    pos = {real_type{3.5} + eps, 1, 0};
    EXPECT_SOFT_EQ(2.25 + eps, calc_distance(px_));
    EXPECT_SOFT_NEAR(0.0 + eps, calc_distance(s_), coarse_eps);

    pos = {real_type{3.5}, 1, 0};
    EXPECT_SOFT_EQ(2.25, calc_distance(px_));
    EXPECT_SOFT_EQ(0.0, calc_distance(s_));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
