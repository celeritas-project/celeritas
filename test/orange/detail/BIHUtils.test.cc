//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BIHUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
TEST(BIHUtilsTest, bbox_vector_union)
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
