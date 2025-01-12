//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHBuilder.hh"

#include <limits>
#include <vector>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "orange/detail/BIHData.hh"
#include "celeritas/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
class BIHBuilderTest : public ::celeritas::test::Test
{
  public:
    using VecFastReal = std::vector<fast_real_type>;

  protected:
    std::vector<FastBBox> bboxes_;
    BIHTreeData<Ownership::value, MemSpace::host> storage_;
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
    using Side = BIHInnerNode::Side;
    using Real3 = FastBBox::Real3;
    constexpr auto inff = std::numeric_limits<fast_real_type>::infinity();

    bboxes_.push_back(FastBBox::from_infinite());
    bboxes_.push_back({{0, 0, 0}, {1.6f, 1, 100}});
    bboxes_.push_back({{1.2f, 0, 0}, {2.8f, 1, 100}});
    bboxes_.push_back({{2.8f, 0, 0}, {5, 1, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

    BIHBuilder build(&storage_);
    auto bih_tree = build(std::move(bboxes_));
    ASSERT_EQ(1, bih_tree.inf_vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);

    // Test bounding box storage
    auto bbox1 = storage_.bboxes[bih_tree.bboxes[LocalVolumeId{2}]];
    EXPECT_VEC_SOFT_EQ(Real3({1.2f, 0, 0}), bbox1.lower());
    EXPECT_VEC_SOFT_EQ(Real3({2.8f, 1, 100}), bbox1.upper());

    // Test nodes
    auto inner_nodes = bih_tree.inner_nodes;
    auto leaf_nodes = bih_tree.leaf_nodes;
    ASSERT_EQ(3, inner_nodes.size());
    ASSERT_EQ(4, leaf_nodes.size());

    // N0, I0
    {
        auto node = storage_.inner_nodes[inner_nodes[0]];
        auto& edges = node.edges;

        ASSERT_FALSE(node.parent);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_SOFT_EQ(2.8f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(0.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{1}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{2}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.8f, inff, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, -inff, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, inff, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N1, I1
    {
        auto node = storage_.inner_nodes[inner_nodes[1]];
        auto& edges = node.edges;

        ASSERT_EQ(BIHNodeId{0}, node.parent);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_SOFT_EQ(1.6f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(1.2f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{3}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{4}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.6f, inff, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.2f, -inff, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.8f, inff, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N2, I2
    {
        auto node = storage_.inner_nodes[inner_nodes[2]];
        auto& edges = node.edges;

        ASSERT_EQ(BIHNodeId{0}, node.parent);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_SOFT_EQ(5.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(2.8f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{5}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{6}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({5.f, inff, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.8f, -inff, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, inff, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N3, L0
    {
        auto node = storage_.leaf_nodes[leaf_nodes[0]];
        ASSERT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{1}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N3, L1
    {
        auto node = storage_.leaf_nodes[leaf_nodes[1]];
        ASSERT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{2}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N5, L2
    {
        auto node = storage_.leaf_nodes[leaf_nodes[2]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(2, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{4}, storage_.local_volume_ids[node.vol_ids[0]]);
        EXPECT_EQ(LocalVolumeId{5}, storage_.local_volume_ids[node.vol_ids[1]]);
    }

    // N6, L3
    {
        auto node = storage_.leaf_nodes[leaf_nodes[3]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{3}, storage_.local_volume_ids[node.vol_ids[0]]);
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
    using Side = BIHInnerNode::Side;
    constexpr auto inff = std::numeric_limits<fast_real_type>::infinity();

    bboxes_.push_back(FastBBox::from_infinite());

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

    BIHBuilder build(&storage_);
    auto bih_tree = build(std::move(bboxes_));
    ASSERT_EQ(1, bih_tree.inf_vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);

    // Test nodes
    auto inner_nodes = bih_tree.inner_nodes;
    auto leaf_nodes = bih_tree.leaf_nodes;
    ASSERT_EQ(11, inner_nodes.size());
    ASSERT_EQ(12, leaf_nodes.size());

    // N0, I0
    {
        auto node = storage_.inner_nodes[inner_nodes[0]];
        auto& edges = node.edges;

        ASSERT_FALSE(node.parent);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_SOFT_EQ(2.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(2.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{1}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{6}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, 2.f, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, 2.f, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, inff, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N1, I1
    {
        auto node = storage_.inner_nodes[inner_nodes[1]];
        auto& edges = node.edges;

        ASSERT_EQ(BIHNodeId{0}, node.parent);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_SOFT_EQ(1.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(1.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{2}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{3}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, -inff, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, 2.f, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N2, I2
    {
        auto node = storage_.inner_nodes[inner_nodes[2]];
        auto& edges = node.edges;

        ASSERT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_SOFT_EQ(1.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(1.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{11}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{12}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 1.f, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, 1.f, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N3, I3
    {
        auto node = storage_.inner_nodes[inner_nodes[3]];
        auto& edges = node.edges;

        ASSERT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_SOFT_EQ(2.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(2.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{4}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{5}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 2.f, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, -inff, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, 2.f, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N4, I4
    {
        auto node = storage_.inner_nodes[inner_nodes[4]];
        auto& edges = node.edges;

        ASSERT_EQ(BIHNodeId{3}, node.parent);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_SOFT_EQ(1.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(1.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{13}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{14}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 1.f, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 1.f, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 2.f, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N5, I5
    {
        auto node = storage_.inner_nodes[inner_nodes[5]];
        auto& edges = node.edges;

        ASSERT_EQ(BIHNodeId{3}, node.parent);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_SOFT_EQ(1.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(1.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{15}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{16}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, 1.f, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 1.f, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, 2.f, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N11, L0
    {
        auto node = storage_.leaf_nodes[leaf_nodes[0]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{1}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N12, L1
    {
        auto node = storage_.leaf_nodes[leaf_nodes[1]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{2}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N13, L2
    {
        auto node = storage_.leaf_nodes[leaf_nodes[2]];
        ASSERT_EQ(BIHNodeId{4}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{5}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N14, L3
    {
        auto node = storage_.leaf_nodes[leaf_nodes[3]];
        ASSERT_EQ(BIHNodeId{4}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{6}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N15, L4
    {
        auto node = storage_.leaf_nodes[leaf_nodes[4]];
        ASSERT_EQ(BIHNodeId{5}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{9}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N16, L5
    {
        auto node = storage_.leaf_nodes[leaf_nodes[5]];
        ASSERT_EQ(BIHNodeId{5}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{10},
                  storage_.local_volume_ids[node.vol_ids[0]]);
    }
}

//---------------------------------------------------------------------------//
// Degenerate, single leaf cases
//---------------------------------------------------------------------------//
//
TEST_F(BIHBuilderTest, single_finite_volume)
{
    bboxes_.push_back({{0, 0, 0}, {1, 1, 1}});

    BIHBuilder build(&storage_);
    auto bih_tree = build(std::move(bboxes_));

    ASSERT_EQ(0, bih_tree.inf_vol_ids.size());
    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());

    auto node = storage_.leaf_nodes[bih_tree.leaf_nodes[0]];
    ASSERT_EQ(BIHNodeId{}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0}, storage_.local_volume_ids[node.vol_ids[0]]);
}

TEST_F(BIHBuilderTest, multiple_nonpartitionable_volumes)
{
    bboxes_.push_back({{0, 0, 0}, {1, 1, 1}});
    bboxes_.push_back({{0, 0, 0}, {1, 1, 1}});

    BIHBuilder build(&storage_);
    auto bih_tree = build(std::move(bboxes_));

    ASSERT_EQ(0, bih_tree.inf_vol_ids.size());
    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());

    auto node = storage_.leaf_nodes[bih_tree.leaf_nodes[0]];
    ASSERT_EQ(BIHNodeId{}, node.parent);
    EXPECT_EQ(2, node.vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0}, storage_.local_volume_ids[node.vol_ids[0]]);
    EXPECT_EQ(LocalVolumeId{1}, storage_.local_volume_ids[node.vol_ids[1]]);
}

TEST_F(BIHBuilderTest, single_infinite_volume)
{
    bboxes_.push_back(FastBBox::from_infinite());

    BIHBuilder build(&storage_);
    auto bih_tree = build(std::move(bboxes_));

    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());
    ASSERT_EQ(1, bih_tree.inf_vol_ids.size());

    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);
}

TEST_F(BIHBuilderTest, multiple_infinite_volumes)
{
    bboxes_.push_back(FastBBox::from_infinite());
    bboxes_.push_back(FastBBox::from_infinite());

    BIHBuilder build(&storage_);
    auto bih_tree = build(std::move(bboxes_));

    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());
    ASSERT_EQ(2, bih_tree.inf_vol_ids.size());

    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);
    EXPECT_EQ(LocalVolumeId{1},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[1]]);
}

TEST_F(BIHBuilderTest, TEST_IF_CELERITAS_DEBUG(semi_finite_volumes))
{
    constexpr auto inff = std::numeric_limits<fast_real_type>::infinity();
    bboxes_.push_back(FastBBox{{0, 0, -inff}, {1, 1, inff}});
    bboxes_.push_back(FastBBox{{1, 0, -inff}, {2, 1, inff}});
    bboxes_.push_back(FastBBox{{2, 0, -inff}, {4, 1, inff}});
    bboxes_.push_back(FastBBox{{4, 0, -inff}, {8, 1, inff}});
    bboxes_.push_back(FastBBox{{0, -inff, -inff}, {1, inff, inff}});
    bboxes_.push_back(FastBBox{{-inff, 0, 0}, {inff, 1, 1}});

    BIHBuilder build(&storage_);
    EXPECT_THROW(build(std::move(bboxes_)), DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
