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
        lvi_storage_;
    Collection<BIHNode, Ownership::value, MemSpace::host, OpaqueId<BIHNode>>
        node_storage_;
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
 */

TEST_F(BIHBuilderTest, fourboxes)
{
    bboxes_.push_back({{0, 0, 0}, {1.6, 1, 100}});
    bboxes_.push_back({{1.2, 0, 0}, {2.8, 1, 100}});
    bboxes_.push_back({{2.8, 0, 0}, {5, 1, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

    BIHBuilder bih(bboxes_, &lvi_storage_, &node_storage_);
    auto bih_params = bih();

    auto nodes = bih_params.nodes;
    EXPECT_EQ(7, nodes.size());

    auto node = node_storage_[nodes[0]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_FALSE(node.parent);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(2.8, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(0, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(1, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(4, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[1]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_EQ(BIHNodeId{0}, node.parent);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(1.6, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(1.2, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(2, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(3, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[2]];
    EXPECT_EQ(BIHNodeId{1}, node.parent);
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(1, lvi_storage_[node.vol_ids[0]].unchecked_get());

    node = node_storage_[nodes[3]];
    EXPECT_EQ(BIHNodeId{1}, node.parent);
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(2, lvi_storage_[node.vol_ids[0]].unchecked_get());

    node = node_storage_[nodes[4]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_EQ(BIHNodeId{0}, node.parent);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(5, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(2.8, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(5, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(6, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[5]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{4}, node.parent);
    EXPECT_EQ(2, node.vol_ids.size());
    EXPECT_EQ(4, lvi_storage_[node.vol_ids[0]].unchecked_get());
    EXPECT_EQ(5, lvi_storage_[node.vol_ids[1]].unchecked_get());

    node = node_storage_[nodes[6]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{4}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(3, lvi_storage_[node.vol_ids[0]].unchecked_get());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
