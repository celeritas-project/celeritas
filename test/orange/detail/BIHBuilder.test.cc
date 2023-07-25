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
    Collection<BIHInnerNode, Ownership::value, MemSpace::host, OpaqueId<BIHInnerNode>>
        inner_node_storage_;
    Collection<BIHLeafNode, Ownership::value, MemSpace::host, OpaqueId<BIHLeafNode>>
        leaf_node_storage_;
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
 * Resultant tree structure in terms of BIHNodeIds (N) and volumes (V):
 *
 *                      ___ N0 ___
 *                    /            \
 *                  N1              N2
 *                 /  \           /    \
 *                N3   N4        N5     N6
 *                V1   V2       V4,V5   V3
 *
 * In terms of BIHInnerNodeIds (I) and BIHLeafNodeIds (L):
 *
 *                      ___ I0 ___
 *                    /            \
 *                  I1              I2
 *                 /  \           /    \
 *                L1   L2        L3     L4
 *                V1   V2       V4,V5   V3
 */
TEST_F(BIHBuilderTest, basic)
{
    bboxes_.push_back({{0, 0, 0}, {1.6, 1, 100}});
    bboxes_.push_back({{1.2, 0, 0}, {2.8, 1, 100}});
    bboxes_.push_back({{2.8, 0, 0}, {5, 1, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

    BIHBuilder bih(
        bboxes_, &lvi_storage_, &inner_node_storage_, &leaf_node_storage_);
    auto bih_params = bih();
    EXPECT_EQ(1, bih_params.inf_volids.size());
    EXPECT_EQ(LocalVolumeId{0}, lvi_storage_[bih_params.inf_volids[0]]);

    auto inner_nodes = bih_params.inner_nodes;
    auto leaf_nodes = bih_params.leaf_nodes;
    EXPECT_EQ(3, inner_nodes.size());
    EXPECT_EQ(4, leaf_nodes.size());

    // N0, I0
    {
        auto node = inner_node_storage_[inner_nodes[0]];
        EXPECT_FALSE(node.parent);
        EXPECT_EQ(Axis{0}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{0},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            2.8, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            0, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(1, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(2, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N1, I1
    {
        auto node = inner_node_storage_[inner_nodes[1]];
        EXPECT_EQ(BIHNodeId{0}, node.parent);
        EXPECT_EQ(Axis{0}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{0},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            1.6, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            1.2, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(3, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(4, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N2, I2
    {
        auto node = inner_node_storage_[inner_nodes[2]];
        EXPECT_EQ(BIHNodeId{0}, node.parent);
        EXPECT_EQ(Axis{0}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{0},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            5, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            2.8, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(5, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(6, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N3, L0
    {
        auto node = leaf_node_storage_[leaf_nodes[0]];
        EXPECT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(1, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }

    // N3, L1
    {
        auto node = leaf_node_storage_[leaf_nodes[1]];
        EXPECT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(2, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }

    // N5, L2
    {
        auto node = leaf_node_storage_[leaf_nodes[2]];
        EXPECT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(2, node.vol_ids.size());
        EXPECT_EQ(4, lvi_storage_[node.vol_ids[0]].unchecked_get());
        EXPECT_EQ(5, lvi_storage_[node.vol_ids[1]].unchecked_get());
    }

    // N6, L3
    {
        auto node = leaf_node_storage_[leaf_nodes[3]];
        EXPECT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(3, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }
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
 * Resultant tree structure in terms of BIHNodeId (N) and volumes (V)
 *
 *                   _______________ N0 ______________
 *                 /                                   \
 *          ___  N1  ___                         ___   N6  ___
 *        /              \                     /                \
 *      N2                N3                 N7                  N8
 *     /   \           /      \             /   \            /       \
 *  N11    N12       N4         N5         N17    N18      N9          N10
 *  V1     V2      /   \      /   \        V3    V4       /  \        /   \
 *                N13  N14   N15   N16                   N19  N20    N21   N22
 *                V5   V6    V9    V10                   V7   V8     V11   V12
 *
 * In terms of BIHInnerNodeIds (I) and BIHLeafNodeIds (L):
 *
 *                   _______________ I0 ______________
 *                 /                                   \
 *          ___  I1  ___                         ___   I6  ___
 *        /              \                     /                \
 *      I2                I3                 I7                 I8
 *     /   \           /      \             /   \            /       \
 *  L0     L1       I4         I5          L6    L7        I9          I10
 *  V1     V2      /   \      /   \        V3    V4       /  \        /   \
 *                L2   L3    L4    L5                    L8   L9     L10   L11
 *                V5   V6    V9    V10                   V7   V8     V11   V12
 *
 *
 *
 * Here, we test only the N1 side for the tree for brevity, as the N6 side is
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

    BIHBuilder bih(
        bboxes_, &lvi_storage_, &inner_node_storage_, &leaf_node_storage_);
    auto bih_params = bih();
    EXPECT_EQ(1, bih_params.inf_volids.size());
    EXPECT_EQ(LocalVolumeId{0}, lvi_storage_[bih_params.inf_volids[0]]);

    auto inner_nodes = bih_params.inner_nodes;
    auto leaf_nodes = bih_params.leaf_nodes;
    EXPECT_EQ(11, inner_nodes.size());
    EXPECT_EQ(12, leaf_nodes.size());

    // N0, I0
    {
        auto node = inner_node_storage_[inner_nodes[0]];
        EXPECT_FALSE(node.parent);
        EXPECT_EQ(Axis{1}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{1},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            2, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            2, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(1, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(6, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N1, I1
    {
        auto node = inner_node_storage_[inner_nodes[1]];
        EXPECT_EQ(BIHNodeId{0}, node.parent);
        EXPECT_EQ(Axis{0}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{0},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(2, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(3, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N2, I2
    {
        auto node = inner_node_storage_[inner_nodes[2]];
        EXPECT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(Axis{1}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{1},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(11, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(12, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N3, I3
    {
        auto node = inner_node_storage_[inner_nodes[3]];
        EXPECT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(Axis{0}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{0},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            2, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            2, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(4, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(5, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N4, I4
    {
        auto node = inner_node_storage_[inner_nodes[4]];
        EXPECT_EQ(BIHNodeId{3}, node.parent);
        EXPECT_EQ(Axis{1}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{1},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(13, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(14, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N5, I5
    {
        auto node = inner_node_storage_[inner_nodes[5]];
        EXPECT_EQ(BIHNodeId{3}, node.parent);
        EXPECT_EQ(Axis{1}, node.bounding_planes[BIHInnerNode::Edge::left].axis);
        EXPECT_EQ(Axis{1},
                  node.bounding_planes[BIHInnerNode::Edge::right].axis);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::left].location);
        EXPECT_SOFT_EQ(
            1, node.bounding_planes[BIHInnerNode::Edge::right].location);
        EXPECT_EQ(15, node.children[BIHInnerNode::Edge::left].unchecked_get());
        EXPECT_EQ(16, node.children[BIHInnerNode::Edge::right].unchecked_get());
    }

    // N11, I0
    {
        auto node = leaf_node_storage_[leaf_nodes[0]];
        EXPECT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(1, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }

    // N12, L1
    {
        auto node = leaf_node_storage_[leaf_nodes[1]];
        EXPECT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(2, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }

    // N13, L2
    {
        auto node = leaf_node_storage_[leaf_nodes[2]];
        EXPECT_EQ(BIHNodeId{4}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(5, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }

    // N14, L3
    {
        auto node = leaf_node_storage_[leaf_nodes[3]];
        EXPECT_EQ(BIHNodeId{4}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(6, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }

    // N15, L4
    {
        auto node = leaf_node_storage_[leaf_nodes[4]];
        EXPECT_EQ(BIHNodeId{5}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(9, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }

    // N16, L5
    {
        auto node = leaf_node_storage_[leaf_nodes[5]];
        EXPECT_EQ(BIHNodeId{5}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(10, lvi_storage_[node.vol_ids[0]].unchecked_get());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
