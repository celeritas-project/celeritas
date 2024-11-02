//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHIntersectingVolFinder.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "orange/detail/BIHBuilder.hh"
#include "orange/detail/BIHData.hh"
#include "celeritas/Types.hh"

#include "celeritas_test.hh"

using BIHBuilder = celeritas::detail::BIHBuilder;
using BIHInnerNode = celeritas::detail::BIHInnerNode;
using BIHLeafNode = celeritas::detail::BIHLeafNode;
using BIHIntersectingVolFinder = celeritas::detail::BIHIntersectingVolFinder;
using Intersection = celeritas::detail::BIHIntersectingVolFinder::Intersection;
using Ray = celeritas::detail::BIHIntersectingVolFinder::Ray;

namespace celeritas
{
namespace test
{
class BIHIntersectingVolFinderTest : public Test
{
  public:
    void SetUp() {}

  protected:
    std::vector<FastBBox> bboxes_;

    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;

    static constexpr BIHIntersectingVolFinder::Intersection
    visit_vol_(LocalVolumeId vol_id)
    {
        return Intersection{vol_id, 3.0};
    };
};

//---------------------------------------------------------------------------//
/* Simple test with partial and fully overlapping bounding boxes.
 *
 *         0    V1    1.6
 *         |--------------|
 *
 *                    1.2   V2    2.8
 *                    |---------------|
 *    y=1 ____________________________________________________
 *        |           |   |           |                      |
 *        |           |   |           |         V3           |
 *    y=0 |___________|___|___________|______________________|
 *        |                                                  |
 *        |             V4, V5 (total overlap)               |
 *   y=-1 |__________________________________________________|
 *
 *        x=0                                                x=5
 */
TEST_F(BIHIntersectingVolFinderTest, basic)
{
    bboxes_.push_back(FastBBox::from_infinite());
    bboxes_.push_back({{0, 0, 0}, {1.6f, 1, 100}});
    bboxes_.push_back({{1.2f, 0, 0}, {2.8f, 1, 100}});
    bboxes_.push_back({{2.8f, 0, 0}, {5, 1, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHIntersectingVolFinder find_volume(bih_tree, ref_storage_);

    Ray ray{{0.8, 0.5, 110.}, {1., 0., 0.}};

    auto x = find_volume(ray, visit_vol_);

    // EXPECT_EQ(LocalVolumeId{1}, find_volume({0.8, 0.5, 30}, valid_volid_));
    // EXPECT_EQ(LocalVolumeId{2}, find_volume({2.0, 0.6, 40}, valid_volid_));
    // EXPECT_EQ(LocalVolumeId{3}, find_volume({2.9, 0.7, 50}, valid_volid_));
    // EXPECT_EQ(LocalVolumeId{4}, find_volume({2.9, -0.7, 50}, valid_volid_));
    // EXPECT_EQ(LocalVolumeId{5}, find_volume({2.9, -0.7, 50}, odd_volid_));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
