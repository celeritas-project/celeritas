//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surfaces/GeneralQuadric.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/GeneralQuadric.hh"

#include "celeritas_test.hh"

using celeritas::GeneralQuadric;
using celeritas::no_intersection;
using celeritas::SignedSense;
using celeritas::SurfaceState;

using celeritas::ipow;
using celeritas::Real3;
using celeritas::real_type;

using Intersections = GeneralQuadric::Intersections;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeneralQuadricTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
/*!
 * This shape is an ellipsoid:
 * - with radii {3, 2, 1},
 * - rotated by 60 degrees about the x axis,
 * - then 30 degrees about z,
 * - and finally translated by {1, 2, 3}.
 */
TEST_F(GeneralQuadricTest, all)
{
    EXPECT_EQ(celeritas::SurfaceType::gq, GeneralQuadric::surface_type());
    EXPECT_EQ(10, GeneralQuadric::Storage::extent);
    EXPECT_EQ(2, GeneralQuadric::Intersections{}.size());

    const Real3 second{10.3125, 22.9375, 15.75};
    const Real3 cross{-21.867141445557, -20.25, 11.69134295109};
    const Real3 first{-11.964745962156, -9.1328585544429, -65.69134295109};
    real_type   zeroth = 77.652245962156;

    GeneralQuadric gq{second, cross, first, zeroth};
    EXPECT_VEC_SOFT_EQ(second, gq.second());
    EXPECT_VEC_SOFT_EQ(cross, gq.cross());
    EXPECT_VEC_SOFT_EQ(first, gq.first());
    EXPECT_SOFT_EQ(zeroth, gq.zeroth());

    EXPECT_EQ(SignedSense::outside, gq.calc_sense({4, 5, 5}));
    EXPECT_EQ(SignedSense::inside, gq.calc_sense({1, 2, 3}));

    const Real3 center{1, 2, 3};
    const Real3 on_surface{3.598076211353292, 3.5, 3};
    const Real3 inward{-0.8660254037844386, -0.5, 0};
    const Real3 outward{0.8660254037844386, 0.5, 0};

    Intersections distances;

    // On surface, inward
    distances = gq.calc_intersections(on_surface, inward, SurfaceState::on);
    EXPECT_SOFT_EQ(6.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // "Not on surface", inward
    distances = gq.calc_intersections(on_surface, inward, SurfaceState::off);
    EXPECT_SOFT_EQ(1e-16, distances[0]);
    EXPECT_SOFT_EQ(6.0, distances[1]);

    // On surface, outward
    distances = gq.calc_intersections(on_surface, outward, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // In center
    distances = gq.calc_intersections(center, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(3.0, distances[1]);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);

    // Outside, hitting both
    const Real3 pos{-2.464101615137754, 0, 3};
    distances = gq.calc_intersections(pos, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(1.0, distances[0]);
    EXPECT_SOFT_EQ(7.0, distances[1]);
}
