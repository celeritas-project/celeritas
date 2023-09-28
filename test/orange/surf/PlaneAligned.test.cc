//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/PlaneAligned.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/PlaneAligned.hh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "corecel/math/Algorithms.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using VecReal = std::vector<real_type>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PlaneAlignedTest : public Test
{
  protected:
    template<class S>
    real_type calc_intersection(S const& surf,
                                Real3 pos,
                                Real3 dir,
                                SurfaceState s = SurfaceState::off)
    {
        static_assert(sizeof(typename S::Intersections) == sizeof(real_type),
                      "Expected plane to have a single intercept");
        return surf.calc_intersections(pos, dir, s)[0];
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PlaneAlignedTest, x_plane)
{
    PlaneX p(1.0);

    EXPECT_EQ((Real3{1.0, 0.0, 0.0}), p.calc_normal(Real3{1.0, 0.1, -4.2}));
    EXPECT_EQ((Real3{1.0, 0.0, 0.0}), p.calc_normal());

    // Test sense
    EXPECT_EQ(SignedSense::outside, p.calc_sense(Real3{1.01, 0.1, 0.0}));
    EXPECT_EQ(SignedSense::inside, p.calc_sense(Real3{0.99, 0.1, 0.0}));

    // Test intersections
    Real3 px{1.0, 0.0, 0.0};
    Real3 mx{-1.0, 0.0, 0.0};
    Real3 py{0.0, 1.0, 0.0};
    Real3 pz{0.0, 0.0, 1.0};

    EXPECT_EQ(no_intersection(),
              calc_intersection(p, {0.9999, 0.0, 0.0}, px, SurfaceState::on));
    EXPECT_EQ(no_intersection(),
              calc_intersection(p, {1.0001, 0.0, 0.0}, mx, SurfaceState::on));
    EXPECT_SOFT_EQ(0.01, calc_intersection(p, {0.99, 0.0, 0.0}, px));
    EXPECT_SOFT_EQ(0.01, calc_intersection(p, {1.01, 0.0, 0.0}, mx));
    EXPECT_SOFT_EQ(no_intersection(),
                   calc_intersection(p, {0.99, 0.0, 0.0}, mx));
    EXPECT_SOFT_EQ(no_intersection(),
                   calc_intersection(p, {1.01, 0.0, 0.0}, px));
    EXPECT_EQ(no_intersection(), calc_intersection(p, {1.01, 0.0, 0.0}, py));
    EXPECT_EQ(no_intersection(), calc_intersection(p, {0.99, 0.0, 0.0}, pz));
}

//---------------------------------------------------------------------------//

TEST_F(PlaneAlignedTest, y_plane)
{
    PlaneY p(-1.0);

    EXPECT_EQ((Real3{0.0, 1.0, 0.0}), p.calc_normal(Real3{1.0, -1.0, -4.2}));

    // Test sense
    EXPECT_EQ(SignedSense::outside, p.calc_sense(Real3{1.01, -0.99, 0.0}));
    EXPECT_EQ(SignedSense::inside, p.calc_sense(Real3{0.99, -1.01, 0.0}));

    // Test intersections
    Real3 py{0.0, 1.0, 0.0};
    Real3 my{0.0, -1.0, 0.0};
    Real3 px{1.0, 0.0, 0.0};

    EXPECT_EQ(no_intersection(),
              calc_intersection(p, {0, -1 + 1e-8, 0.0}, py, SurfaceState::on));
    EXPECT_SOFT_EQ(0.01, calc_intersection(p, {0.0, -1.01, 0.0}, py));
    EXPECT_SOFT_EQ(no_intersection(),
                   calc_intersection(p, {0.0, -1.1, 0.0}, my));
    EXPECT_EQ(no_intersection(), calc_intersection(p, {-1.01, 1.0, 0.0}, px));
}

//---------------------------------------------------------------------------//

TEST_F(PlaneAlignedTest, plane_z)
{
    PlaneZ p(0.0);

    EXPECT_EQ((Real3{0.0, 0.0, 1.0}), p.calc_normal(Real3{1.0, 0.1, 0.0}));

    // Test sense
    EXPECT_EQ(SignedSense::outside, p.calc_sense(Real3{1.01, 0.1, 0.01}));
    EXPECT_EQ(SignedSense::inside, p.calc_sense(Real3{0.99, 0.1, -0.01}));

    // Test intersections
    Real3 pz({0.0, 0.0, 1.0});
    Real3 mz({0.0, 0.0, -1.0});
    Real3 px({1.0, 0.0, 0.0});

    EXPECT_EQ(no_intersection(),
              calc_intersection(p, {0.0, 0.0, -1e-8}, pz, SurfaceState::on));
    EXPECT_SOFT_EQ(0.01, calc_intersection(p, {0.0, 0.0, -0.01}, pz));
    EXPECT_SOFT_EQ(0.01, calc_intersection(p, {0.0, 0.0, 0.01}, mz));
    EXPECT_EQ(no_intersection(), calc_intersection(p, {-1.01, 0.0, 0.0}, px));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
