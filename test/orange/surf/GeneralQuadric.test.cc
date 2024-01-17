//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/GeneralQuadric.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/GeneralQuadric.hh"

#include "corecel/math/Algorithms.hh"
#include "orange/surf/SimpleQuadric.hh"

#include "SurfaceTestUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using Intersections = GeneralQuadric::Intersections;

//---------------------------------------------------------------------------//
TEST(GeneralQuadricTest, construction)
{
    GeneralQuadric gq{SimpleQuadric{{ipow<2>(2.5) * ipow<2>(0.3),
                                     ipow<2>(1.0) * ipow<2>(0.3),
                                     ipow<2>(1.0) * ipow<2>(2.5)},
                                    {0, 0, 0},
                                    -1 * ipow<2>(2.5) * ipow<2>(0.3)}};

    auto distances
        = calc_intersections(gq, {-2.5, 0, 0}, {1, 0, 0}, SurfaceState::off);
    EXPECT_SOFT_EQ(1.5, distances[0]);
    EXPECT_SOFT_EQ(1.5 + 2.0, distances[1]);
    distances
        = calc_intersections(gq, {0, 2.5, 0}, {0, -1, 0}, SurfaceState::on);
    EXPECT_SOFT_EQ(5.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
    distances = calc_intersections(gq, {0, 0, 0}, {0, 0, 1}, SurfaceState::off);
    EXPECT_SOFT_EQ(0.3, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);
}

//---------------------------------------------------------------------------//
/*!
 * This shape is an ellipsoid:
 * - with radii {3, 2, 1},
 * - rotated by 60 degrees about the x axis,
 * - then 30 degrees about z,
 * - and finally translated by {1, 2, 3}.
 */
TEST(GeneralQuadricTest, all)
{
    EXPECT_EQ(SurfaceType::gq, GeneralQuadric::surface_type());
    EXPECT_EQ(10, GeneralQuadric::StorageSpan::extent);
    EXPECT_EQ(2, GeneralQuadric::Intersections{}.size());

    Real3 const second{10.3125, 22.9375, 15.75};
    Real3 const cross{-21.867141445557, -20.25, 11.69134295109};
    Real3 const first{-11.964745962156, -9.1328585544429, -65.69134295109};
    real_type zeroth = 77.652245962156;

    GeneralQuadric gq{second, cross, first, zeroth};
    EXPECT_VEC_SOFT_EQ(second, gq.second());
    EXPECT_VEC_SOFT_EQ(cross, gq.cross());
    EXPECT_VEC_SOFT_EQ(first, gq.first());
    EXPECT_SOFT_EQ(zeroth, gq.zeroth());

    EXPECT_EQ(SignedSense::outside, gq.calc_sense({4, 5, 5}));
    EXPECT_EQ(SignedSense::inside, gq.calc_sense({1, 2, 3}));

    Real3 const center{1, 2, 3};
    Real3 const on_surface{3.598076211353292, 3.5, 3};
    Real3 const inward{-0.8660254037844386, -0.5, 0};
    Real3 const outward{0.8660254037844386, 0.5, 0};

    Intersections distances;

    // On surface, inward
    distances = calc_intersections(gq, on_surface, inward, SurfaceState::on);
    EXPECT_SOFT_EQ(6.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // "Not on surface", inward
    distances = calc_intersections(gq, on_surface, inward, SurfaceState::off);
    if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_EQ(1e-16, distances[0]);
        EXPECT_SOFT_EQ(6.0, distances[1]);
    }
    else if (distances[1] == no_intersection())
    {
        // x86 hardware
        EXPECT_SOFT_EQ(6.0f, distances[0]);
    }
    else
    {
        // Apple Silicon
        EXPECT_SOFT_EQ(1e-7f, distances[0]);
        EXPECT_SOFT_EQ(6.0f, distances[1]);
    }

    // On surface, outward
    distances = calc_intersections(gq, on_surface, outward, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // In center
    distances = calc_intersections(gq, center, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(3.0, distances[0]);
    EXPECT_SOFT_EQ(no_intersection(), distances[1]);

    // Outside, hitting both
    Real3 const pos{-2.464101615137754, 0, 3};
    distances = calc_intersections(gq, pos, outward, SurfaceState::off);
    EXPECT_SOFT_EQ(1.0, distances[0]);
    EXPECT_SOFT_EQ(7.0, distances[1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
