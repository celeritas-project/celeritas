//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHBuilder.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Types.hh"

#include "celeritas_test.hh"

using BIHBuilder = celeritas::detail::BIHBuilder;

namespace celeritas
{
namespace test
{
class BIHBuilderTest : public Test
{
  public:
    void SetUp()
    {
        auto inf = std::numeric_limits<double>::infinity();
        bboxes_.push_back({{-inf, -inf, -inf}, {inf, inf, inf}});
    }

  protected:
    std::vector<BoundingBox> bboxes_;
    Collection<LocalVolumeId, Ownership::value, MemSpace::host, OpaqueId<LocalVolumeId>>
        storage_;
};

//---------------------------------------------------------------------------//
//
/* Simple 4 bbox test. The Z dimensions extends from [0, 100], and it therefore
 * the longest dimension. However, all bboxes have the same Z-center, so the
 * first split is X=
 *
 *         0    vol 1    1.6
 *         |--------------|
 *
 *                    1.2   vol 2    2.8
 *                    |---------------|
 *    y=1 ____________________________________________________
 *        |           |   |           |                      |
 *        |           |   |           |      vol 3           |
 *    y=0 |___________|___|___________|______________________|
 *        |                                                  |
 *        |          vols 4, 5 (total overlap)               |
 *   y=-1 |__________________________________________________|
 *
 *        x=0                                                x=5
 *
 * Volumes within Resultant tree structure:
 *
 *                      root
 *                   /        \
 *                 / \      /   \
 *                1   2    4, 5   3
 *
 * 0.8     2     3.9
 *           2.5
 */

TEST_F(BIHBuilderTest, fourboxes)
{
    bboxes_.push_back({{0, 0, 0}, {1.6, 1, 100}});
    bboxes_.push_back({{1.2, 0, 0}, {2.8, 1, 100}});
    bboxes_.push_back({{2.8, 0, 0}, {5, 1, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

    BIHBuilder bih(bboxes_, &storage_);
    auto nodes = bih();

    EXPECT_EQ(7, nodes.size());

    EXPECT_TRUE(nodes[0].is_inner());
    EXPECT_SOFT_EQ(2.8, nodes[0].partitions[BIHNode::Edge::left]);
    EXPECT_SOFT_EQ(0, nodes[0].partitions[BIHNode::Edge::right]);
    EXPECT_EQ(1, nodes[0].children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(4, nodes[0].children[BIHNode::Edge::right].unchecked_get());

    EXPECT_TRUE(nodes[1].is_inner());
    EXPECT_SOFT_EQ(1.6, nodes[1].partitions[BIHNode::Edge::left]);
    EXPECT_SOFT_EQ(1.2, nodes[1].partitions[BIHNode::Edge::right]);
    EXPECT_EQ(2, nodes[1].children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(3, nodes[1].children[BIHNode::Edge::right].unchecked_get());

    EXPECT_TRUE(nodes[2].is_leaf());
    EXPECT_EQ(1, nodes[2].vol_ids.size());
    EXPECT_EQ(1, storage_[nodes[2].vol_ids[0]].unchecked_get());

    EXPECT_TRUE(nodes[3].is_leaf());
    EXPECT_EQ(1, nodes[3].vol_ids.size());
    EXPECT_EQ(2, storage_[nodes[3].vol_ids[0]].unchecked_get());

    EXPECT_TRUE(nodes[4].is_inner());
    EXPECT_SOFT_EQ(5, nodes[4].partitions[BIHNode::Edge::left]);
    EXPECT_SOFT_EQ(2.8, nodes[4].partitions[BIHNode::Edge::right]);
    EXPECT_EQ(5, nodes[4].children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(6, nodes[4].children[BIHNode::Edge::right].unchecked_get());

    EXPECT_TRUE(nodes[5].is_leaf());
    EXPECT_EQ(2, nodes[5].vol_ids.size());
    EXPECT_EQ(4, storage_[nodes[5].vol_ids[0]].unchecked_get());
    EXPECT_EQ(5, storage_[nodes[5].vol_ids[1]].unchecked_get());

    EXPECT_TRUE(nodes[6].is_leaf());
    EXPECT_EQ(1, nodes[6].vol_ids.size());
    EXPECT_EQ(3, storage_[nodes[6].vol_ids[0]].unchecked_get());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
