//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

TEST_F(SurfaceFunctorsTest, num_intersections)
{
    EXPECT_EQ(1, visit_surface_type(NumIntersections{}, SurfaceType::px));
    EXPECT_EQ(2, visit_surface_type(NumIntersections{}, SurfaceType::sc));
}

//---------------------------------------------------------------------------//

TEST_F(SurfaceFunctorsTest, calc_safety_distance)
{
    Real3 pos;
    CalcSafetyDistance calc_distance{pos};

    real_type eps = 1e-4;
    pos = {1.25_r + eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(s_));

    pos = {1.25_r, 1, 0};
    EXPECT_SOFT_EQ(0, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25, calc_distance(s_));

    pos = {1.25_r - eps, 1, 0};
    EXPECT_SOFT_EQ(eps, calc_distance(px_));
    EXPECT_SOFT_EQ(0.25 - eps, calc_distance(s_));

    pos = {1_r - eps, 1, 0};
    EXPECT_SOFT_EQ(0.25 + eps, calc_distance(px_));
    EXPECT_SOFT_EQ(eps, calc_distance(s_));

    pos = {3.5_r + eps, 1, 0};
    EXPECT_SOFT_EQ(2.25 + eps, calc_distance(px_));
    EXPECT_SOFT_NEAR(0.0 + eps, calc_distance(s_), coarse_eps);

    pos = {3.5_r, 1, 0};
    EXPECT_SOFT_EQ(2.25, calc_distance(px_));
    EXPECT_SOFT_EQ(0.0, calc_distance(s_));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
