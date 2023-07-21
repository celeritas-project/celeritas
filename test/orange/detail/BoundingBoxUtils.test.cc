//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BoundingBoxUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BoundingBoxUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{

TEST(BoundingBoxUtilsTest, is_infinite)
{
    auto max_real = std::numeric_limits<real_type>::max();
    auto inf_real = std::numeric_limits<real_type>::infinity();

    BoundingBox bbox1 = {{0, 0, 0}, {1, 1, 1}};
    EXPECT_FALSE(is_infinite(bbox1));

    BoundingBox bbox2 = {{0, 0, 0}, {max_real, inf_real, max_real}};
    EXPECT_FALSE(is_infinite(bbox2));

    BoundingBox bbox3
        = {{-max_real, -inf_real, -max_real}, {max_real, inf_real, max_real}};
    EXPECT_TRUE(is_infinite(bbox3));
}

TEST(BoundingBoxUtilsTest, sort_axes)
{
    BoundingBox bbox1 = {{-5, -2, -100}, {6, 1, 1}};
    EXPECT_EQ(std::vector<Axis>({Axis::z, Axis::x, Axis::y}), sort_axes(bbox1));

    BoundingBox bbox2 = {{-5, -13, -10}, {6, 10, 2}};
    EXPECT_EQ(std::vector<Axis>({Axis::y, Axis::z, Axis::x}), sort_axes(bbox2));

    BoundingBox bbox3 = {{-50, -13, -10}, {6, 10, 2}};
    EXPECT_EQ(std::vector<Axis>({Axis::x, Axis::y, Axis::z}), sort_axes(bbox3));
}

TEST(BoundingBoxUtilsTest, center)
{
    BoundingBox bbox = {{-10, -20, -30}, {1, 2, 3}};
    EXPECT_VEC_SOFT_EQ(Real3({-4.5, -9, -13.5}), center(bbox));
}

TEST(BoundingBoxUtilsTest, bbox_union)
{
    BoundingBox bbox1 = {{-10, -20, -30}, {10, 2, 3}};
    BoundingBox bbox2 = {{-15, -9, -33}, {1, 2, 10}};

    auto bbox3 = bbox_union(bbox1, bbox2);

    EXPECT_VEC_SOFT_EQ(Real3({-15, -20, -33}), bbox3.lower());
    EXPECT_VEC_SOFT_EQ(Real3({10, 2, 10}), bbox3.upper());
}

TEST(BoundingBoxUtilsTest, bbox_vector_union)
{
    BoundingBox bbox1 = {{-10, -20, -30}, {10, 2, 3}};
    BoundingBox bbox2 = {{-15, -9, -33}, {1, 2, 10}};
    BoundingBox bbox3 = {{-15, -9, -34}, {1, 2, 10}};

    std::vector<BoundingBox> bboxes{bbox1, bbox2, bbox3};

    std::vector<LocalVolumeId> ids_subset{LocalVolumeId(0), LocalVolumeId(1)};
    std::vector<LocalVolumeId> ids_all{
        LocalVolumeId(0), LocalVolumeId(1), LocalVolumeId(2)};

    auto bbox4 = bbox_union(bboxes, ids_subset);
    EXPECT_VEC_SOFT_EQ(Real3({-15, -20, -33}), bbox4.lower());
    EXPECT_VEC_SOFT_EQ(Real3({10, 2, 10}), bbox4.upper());

    auto bbox5 = bbox_union(bboxes, ids_all);
    EXPECT_VEC_SOFT_EQ(Real3({-15, -20, -34}), bbox5.lower());
    EXPECT_VEC_SOFT_EQ(Real3({10, 2, 10}), bbox5.upper());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
