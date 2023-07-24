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
 *
 * Resultant tree structure with numbered nodes (N) and volumes (V)
 *
 *                      ___ N0 ___
 *                    /            \
 *                  N1              N4
 *                 /  \           /    \
 *                N2   N3        N5     N6
 *                V1   V2       V4,V5   V3
 */
TEST_F(BIHBuilderTest, basic)
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
/* Test a 3x4 grid of non-overlapping cuboids.
 *
 *                4 _______________
 *                  | V4 | V8 | V12|
 *                3 |____|____|____|
 *                  | V3 | V7 | V11|
 *            y   2 |____|____|____|
 *                  | V2 | V6 | V10|
 *                1 |____|____|____|
 *                  | V1 | V5 | V9 |
 *                0 |____|____|____|
 *                  0    1    2    3
 *                          x
 *
 * Resultant tree structure with numbered nodes (N) and volumes (V)
 *
 *                  _____________  N0 _____________
 *                /                                 \
 *        ___   N1  ___                             N12
 *      /               \                           ...
 *    N2                 N5                    analogous to N1
 *   /   \            /      \                      ...
 *  N3    N4        N6         N9
 *  V1    V2       /  \      /   \
 *                N7   N8   N10   N11
 *                V5   V6   V9    V10
 *
 * Here, we test only the N1 side for the tree for brevity, as the N1 side is
 * directly analogus.
 */
TEST_F(BIHBuilderTest, grid)
{
    bboxes_.push_back({{0, 0, 0}, {1, 1, 100}});
    bboxes_.push_back({{0, 1, 0}, {1, 2, 100}});
    bboxes_.push_back({{0, 2, 0}, {1, 3, 100}});
    bboxes_.push_back({{0, 3, 0}, {1, 4, 100}});

    bboxes_.push_back({{1, 0, 0}, {2, 1, 100}});
    bboxes_.push_back({{1, 1, 0}, {2, 2, 100}});
    bboxes_.push_back({{1, 2, 0}, {2, 3, 100}});
    bboxes_.push_back({{1, 3, 0}, {2, 4, 100}});

    bboxes_.push_back({{2, 0, 0}, {3, 1, 100}});
    bboxes_.push_back({{2, 1, 0}, {3, 2, 100}});
    bboxes_.push_back({{2, 2, 0}, {3, 3, 100}});
    bboxes_.push_back({{2, 3, 0}, {3, 4, 100}});

    BIHBuilder bih(bboxes_, &lvi_storage_, &node_storage_);
    auto bih_params = bih();

    auto nodes = bih_params.nodes;
    EXPECT_EQ(23, nodes.size());

    auto node = node_storage_[nodes[0]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_FALSE(node.parent);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(2, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(2, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(1, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(12, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[1]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_EQ(BIHNodeId{0}, node.parent);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(2, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(5, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[2]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_EQ(BIHNodeId{1}, node.parent);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(3, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(4, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[3]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{2}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(1, lvi_storage_[node.vol_ids[0]].unchecked_get());

    node = node_storage_[nodes[4]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{2}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(2, lvi_storage_[node.vol_ids[0]].unchecked_get());

    node = node_storage_[nodes[5]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_EQ(BIHNodeId{1}, node.parent);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{0}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(2, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(2, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(6, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(9, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[6]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_EQ(BIHNodeId{5}, node.parent);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(7, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(8, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[7]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{6}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(5, lvi_storage_[node.vol_ids[0]].unchecked_get());

    node = node_storage_[nodes[8]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{6}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(6, lvi_storage_[node.vol_ids[0]].unchecked_get());

    node = node_storage_[nodes[9]];
    EXPECT_TRUE(node.is_inner());
    EXPECT_EQ(BIHNodeId{5}, node.parent);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::left].axis);
    EXPECT_EQ(Axis{1}, node.bounding_planes[BIHNode::Edge::right].axis);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::left].location);
    EXPECT_SOFT_EQ(1, node.bounding_planes[BIHNode::Edge::right].location);
    EXPECT_EQ(10, node.children[BIHNode::Edge::left].unchecked_get());
    EXPECT_EQ(11, node.children[BIHNode::Edge::right].unchecked_get());

    node = node_storage_[nodes[10]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{9}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(9, lvi_storage_[node.vol_ids[0]].unchecked_get());

    node = node_storage_[nodes[11]];
    EXPECT_TRUE(node.is_leaf());
    EXPECT_EQ(BIHNodeId{9}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(10, lvi_storage_[node.vol_ids[0]].unchecked_get());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
