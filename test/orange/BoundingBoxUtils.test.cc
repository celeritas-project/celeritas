//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BoundingBoxUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/BoundingBoxUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
TEST(BoundingBoxUtilsTest, is_inside)
{
    BBox bbox1 = {{-5, -2, -100}, {6, 1, 1}};
    EXPECT_TRUE(is_inside(bbox1, Real3{-4, 0, 0}));
    EXPECT_TRUE(is_inside(bbox1, Real3{-4.9, -1.9, -99.9}));
    EXPECT_FALSE(is_inside(bbox1, Real3{-6, 0, 0}));
    EXPECT_FALSE(is_inside(bbox1, Real3{-5.1, -2.1, -101.1}));
}

TEST(BoundingBoxUtilsTest, is_infinite)
{
    auto inf_real = std::numeric_limits<real_type>::infinity();

    BBox bbox1 = {{0, 0, 0}, {1, 1, 1}};
    EXPECT_FALSE(is_infinite(bbox1));

    BBox bbox2 = {{0, 0, 0}, {inf_real, inf_real, inf_real}};
    EXPECT_FALSE(is_infinite(bbox2));

    BBox bbox3
        = {{-inf_real, -inf_real, -inf_real}, {inf_real, inf_real, inf_real}};
    EXPECT_TRUE(is_infinite(bbox3));
}

TEST(BoundingBoxUtilsTest, center)
{
    BBox bbox = {{-10, -20, -30}, {1, 2, 3}};
    EXPECT_VEC_SOFT_EQ(Real3({-4.5, -9, -13.5}), calc_center(bbox));
}

TEST(BoundingBoxUtilsTest, surface_area)
{
    BBox bbox = {{-1, -2, -3}, {6, 4, 5}};
    EXPECT_SOFT_EQ(2 * (7 * 6 + 7 * 8 + 6 * 8), calc_surface_area(bbox));
}

TEST(BoundingBoxUtilsTest, bbox_union)
{
    BBox bbox1 = {{-10, -20, -30}, {10, 2, 3}};
    BBox bbox2 = {{-15, -9, -33}, {1, 2, 10}};

    auto bbox3 = calc_union(bbox1, bbox2);

    EXPECT_VEC_SOFT_EQ(Real3({-15, -20, -33}), bbox3.lower());
    EXPECT_VEC_SOFT_EQ(Real3({10, 2, 10}), bbox3.upper());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
