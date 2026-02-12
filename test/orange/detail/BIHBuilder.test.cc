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
#include "corecel/data/ParamsDataStore.hh"
#include "geocel/Types.hh"
#include "orange/detail/BIHData.hh"

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
    using VecFastBbox = BIHBuilder::VecBBox;

  protected:
    static constexpr auto inff
        = std::numeric_limits<fast_real_type>::infinity();

    BIHBuilder::SetLocalVolId implicit_vol_ids_;
    BIHTreeData<Ownership::value, MemSpace::host> storage_;
};

//---------------------------------------------------------------------------//
/* Simple test with partial and fully overlapping bounding boxes.
 * \verbatim
           0    V1    1.6
           |--------------|

                      1.2   V2    2.8
                      |---------------|
      y=1 ____________________________________________________
          |           |   |           |                      |
          |           |   |           |         V3           |
      y=0 |___________|___|___________|______________________|
          |                                                  |
          |             V4, V5 (total overlap)               |
     y=-1 |__________________________________________________|

          x=0                                                x=5
   \endverbatim
 * Resultant tree structure in terms of BIHNodeIds (N) and volumes (V):
 * \verbatim
                        ___ N0 ___
                      /            \
                    N1              N2
                   /  \           /    \
                  N3   N4        N5     N6
                  V1   V2       V4,V5   V3
   \endverbatim
 * In terms of BIHInnerNodeIds (I) and BIHLeafNodeIds (L):
 * \verbatim
                        ___ I0 ___
                      /            \
                    I1              I2
                   /  \           /    \
                  L1   L2        L3     L4
                  V1   V2       V4,V5   V3
   \endverbatim
 */
TEST_F(BIHBuilderTest, basic)
{
    using Side = BIHInnerNode::Side;
    using Real3 = FastBBox::Real3;

    VecFastBbox bboxes = {
        FastBBox::from_infinite(),
        {{0, 0, 0}, {1.6f, 1, 100}},
        {{1.2f, 0, 0}, {2.8f, 1, 100}},
        {{2.8f, 0, 0}, {5, 1, 100}},
        {{0, -1, 0}, {5, 0, 100}},
        {{0, -1, 0}, {5, 0, 100}},
    };

    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);

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

    // Metadata
    {
        auto const& md = bih_tree.metadata;
        EXPECT_EQ(5, md.num_finite_bboxes);
        EXPECT_EQ(1, md.num_infinite_bboxes);
        EXPECT_EQ(3, md.depth);
    }
}

//---------------------------------------------------------------------------//
// Grid geometry tests
//---------------------------------------------------------------------------//
/* Test a 3x4 grid of non-overlapping cuboids.
 * \verbatim
                  4 _______________
                    | V4 | V8 | V12|
                  3 |____|____|____|
                    | V3 | V7 | V11|
              y   2 |____|____|____|
                    | V2 | V6 | V10|
                  1 |____|____|____|
                    | V1 | V5 | V9 |
                  0 |____|____|____|
                    0    1    2    3
                            x
   \endverbatim
 */
class GridTest : public BIHBuilderTest
{
  protected:
    /// TYPES ///
    using Side = BIHInnerNode::Side;

    /// METHODS ///
    void SetUp() override
    {
        bboxes = {FastBBox::from_infinite()};
        for (auto i : range(3))
        {
            for (auto j : range(4))
            {
                auto x = static_cast<fast_real_type>(i);
                auto y = static_cast<fast_real_type>(j);
                bboxes.push_back({{x, y, 0}, {x + 1, y + 1, 100}});
            }
        }
    }

    /// DATA ///
    VecFastBbox bboxes;
};

//---------------------------------------------------------------------------//
/* Test with max_leaf_size = 1 and the default depth limit (large enough to not
 * affect BIH construction here). The resultant tree structure in terms of
 * BIHNodeId (N) and volumes (V) is:
 * \verbatim
                     _______________ N0 ______________
                   /                                   \
            ___  N1  ___                         ___   N6  ___
          /              \                     /                \
        N2                N3                 N7                  N8
       /   \           /      \             /   \            /       \
    N11    N12       N4         N5         N17    N18      N9          N10
    V1     V2      /   \      /   \        V3    V4       /  \        /   \
                  N13  N14   N15   N16                   N19  N20    N21   N22
                  V5   V6    V9    V10                   V7   V8     V11   V12
   \endverbatim
 * In terms of BIHInnerNodeIds (I) and BIHLeafNodeIds (L):
 * \verbatim
                     _______________ I0 ______________
                   /                                   \
            ___  I1  ___                         ___   I6  ___
          /              \                     /                \
        I2                I3                 I7                 I8
       /   \           /      \             /   \            /       \
    L0     L1       I4         I5          L6    L7        I9          I10
    V1     V2      /   \      /   \        V3    V4       /  \        /   \
                  L2   L3    L4    L5                    L8   L9     L10   L11
                  V5   V6    V9    V10                   V7   V8     V11   V12
   \endverbatim
 * Here, we test only the N1 side for the tree for brevity, as the N6 side is
 * directly analogous.
 */
TEST_F(GridTest, basic)
{
    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);
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

    // Metadata
    {
        auto const& md = bih_tree.metadata;
        EXPECT_EQ(12, md.num_finite_bboxes);
        EXPECT_EQ(1, md.num_infinite_bboxes);
        EXPECT_EQ(5, md.depth);
    }
}

//---------------------------------------------------------------------------//
/* Test with max_leaf_size = 4 and the default depth limit (large enough to not
 * affect BIH construction here). The resultant tree structure in terms of
 * BIHNodeId (N) and volumes (V) is:
 * \verbatim
                     _______________ N0 ______________
                   /                                   \
            ___  N1  ___                         ___   N2  ___
          /              \                     /                \
        N3                N4                 N5                  N6
      V1, V2        V5, V6, V9, V10        V3, V4          V7, V8, V11, V12
   \endverbatim
 * In terms of BIHInnerNodeIds (I) and BIHLeafNodeIds (L):
 * \verbatim

                     _______________ I0 ______________
                   /                                   \
            ___  I1  ___                         ___   I2  ___
          /              \                     /                \
        L0                L1                 L2                  L3
      V1, V2        V5, V6, V9, V10        V3, V4          V7, V8, V11, V12
   \endverbatim
 */
TEST_F(GridTest, max_leaf_size)
{
    BIHBuilder build(&storage_, BIHBuilder::Input{4});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);
    ASSERT_EQ(1, bih_tree.inf_vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);

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
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_EQ(Axis{1}, node.axis);
        EXPECT_SOFT_EQ(2.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(2.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{1}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{2}, edges[Side::right].child);

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
        EXPECT_EQ(BIHNodeId{3}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{4}, edges[Side::right].child);

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

        ASSERT_EQ(BIHNodeId{0}, node.parent);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_EQ(Axis{0}, node.axis);
        EXPECT_SOFT_EQ(1.f, edges[Side::left].bounding_plane_pos);
        EXPECT_SOFT_EQ(1.f, edges[Side::right].bounding_plane_pos);
        EXPECT_EQ(BIHNodeId{5}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{6}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({-inff, 2.f, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, inff, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, inff, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N3, L0
    {
        auto node = storage_.leaf_nodes[leaf_nodes[0]];
        ASSERT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(2, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{1}, storage_.local_volume_ids[node.vol_ids[0]]);
        EXPECT_EQ(LocalVolumeId{2}, storage_.local_volume_ids[node.vol_ids[1]]);
    }

    // N4, L1
    {
        auto node = storage_.leaf_nodes[leaf_nodes[1]];
        ASSERT_EQ(BIHNodeId{1}, node.parent);
        EXPECT_EQ(4, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{5}, storage_.local_volume_ids[node.vol_ids[0]]);
        EXPECT_EQ(LocalVolumeId{6}, storage_.local_volume_ids[node.vol_ids[1]]);
        EXPECT_EQ(LocalVolumeId{9}, storage_.local_volume_ids[node.vol_ids[2]]);
        EXPECT_EQ(LocalVolumeId{10},
                  storage_.local_volume_ids[node.vol_ids[3]]);
    }

    // N5, L2
    {
        auto node = storage_.leaf_nodes[leaf_nodes[2]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(2, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{3}, storage_.local_volume_ids[node.vol_ids[0]]);
        EXPECT_EQ(LocalVolumeId{4}, storage_.local_volume_ids[node.vol_ids[1]]);
    }

    // N6, L3
    {
        auto node = storage_.leaf_nodes[leaf_nodes[3]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(4, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{7}, storage_.local_volume_ids[node.vol_ids[0]]);
        EXPECT_EQ(LocalVolumeId{8}, storage_.local_volume_ids[node.vol_ids[1]]);
        EXPECT_EQ(LocalVolumeId{11},
                  storage_.local_volume_ids[node.vol_ids[2]]);
        EXPECT_EQ(LocalVolumeId{12},
                  storage_.local_volume_ids[node.vol_ids[3]]);
    }

    // Metadata
    {
        auto const& md = bih_tree.metadata;
        EXPECT_EQ(12, md.num_finite_bboxes);
        EXPECT_EQ(1, md.num_infinite_bboxes);
        EXPECT_EQ(3, md.depth);
    }
}

//---------------------------------------------------------------------------//
/* Test with max_leaf_size = 1 and depth_limit = 4, the later of which causes
 * the tree to be less deep than it otherwise would. The resultant tree
 * structure in terms of BIHNodeId (N) and volumes (V) is:
 * \verbatim
                     _______________ N0 ______________
                   /                                   \
            ___  N1  ___                         ___   N4  ___
          /              \                     /               \
        N2                N3                 N5                N6
       /  \             /    \             /    \             /   \
    N7     N8         N9      N10       N11      N12       N13     N14
    V1     V2      V5, V6   V9, V10     V3       V4      V7, V8   V11, V12
   \endverbatim
 * In terms of BIHInnerNodeIds (I) and BIHLeafNodeIds (L):
 * \verbatim
                     _______________ I0 ______________
                   /                                   \
            ___  I1  ___                         ___   I4  ___
          /              \                     /               \
        I2                I3                 I5                 I6
       /  \             /    \             /    \             /    \
    L0     L1         L2      L3         L4      L5         L6      L7
    V1     V2      V5, V6   V9, V10     V3       V4      V7, V8   V11, V12
   \endverbatim
 * Here, we test only the N1 side for the tree for brevity, as the N4 side is
 * directly analogous.
 */
TEST_F(GridTest, depth_limit)
{
    BIHBuilder build(&storage_, BIHBuilder::Input{1, 4});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);
    ASSERT_EQ(1, bih_tree.inf_vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);

    // Test nodes
    auto inner_nodes = bih_tree.inner_nodes;
    auto leaf_nodes = bih_tree.leaf_nodes;
    ASSERT_EQ(7, inner_nodes.size());
    ASSERT_EQ(8, leaf_nodes.size());

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
        EXPECT_EQ(BIHNodeId{4}, edges[Side::right].child);

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
        EXPECT_EQ(BIHNodeId{7}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{8}, edges[Side::right].child);

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
        EXPECT_EQ(BIHNodeId{9}, edges[Side::left].child);
        EXPECT_EQ(BIHNodeId{10}, edges[Side::right].child);

        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, -inff, -inff}),
                           edges[Side::left].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 2.f, inff}),
                           edges[Side::left].bbox.upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, -inff, -inff}),
                           edges[Side::right].bbox.lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({inff, 2.f, inff}),
                           edges[Side::right].bbox.upper());
    }

    // N7, L0
    {
        auto node = storage_.leaf_nodes[leaf_nodes[0]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{1}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N8, L1
    {
        auto node = storage_.leaf_nodes[leaf_nodes[1]];
        ASSERT_EQ(BIHNodeId{2}, node.parent);
        EXPECT_EQ(1, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{2}, storage_.local_volume_ids[node.vol_ids[0]]);
    }

    // N9, L2
    {
        auto node = storage_.leaf_nodes[leaf_nodes[2]];
        ASSERT_EQ(BIHNodeId{3}, node.parent);
        EXPECT_EQ(2, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{5}, storage_.local_volume_ids[node.vol_ids[0]]);
        EXPECT_EQ(LocalVolumeId{6}, storage_.local_volume_ids[node.vol_ids[1]]);
    }

    // N10, L3
    {
        auto node = storage_.leaf_nodes[leaf_nodes[3]];
        ASSERT_EQ(BIHNodeId{3}, node.parent);
        EXPECT_EQ(2, node.vol_ids.size());
        EXPECT_EQ(LocalVolumeId{9}, storage_.local_volume_ids[node.vol_ids[0]]);
        EXPECT_EQ(LocalVolumeId{10},
                  storage_.local_volume_ids[node.vol_ids[1]]);
    }

    // Metadata
    {
        auto const& md = bih_tree.metadata;
        EXPECT_EQ(12, md.num_finite_bboxes);
        EXPECT_EQ(1, md.num_infinite_bboxes);
        EXPECT_EQ(4, md.depth);
    }
}

//---------------------------------------------------------------------------//
// Degenerate, single leaf cases
//---------------------------------------------------------------------------//
//
TEST_F(BIHBuilderTest, single_finite_volume)
{
    VecFastBbox bboxes = {{{0, 0, 0}, {1, 1, 1}}};

    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);

    ASSERT_EQ(0, bih_tree.inf_vol_ids.size());
    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());

    auto node = storage_.leaf_nodes[bih_tree.leaf_nodes[0]];
    ASSERT_EQ(BIHNodeId{}, node.parent);
    EXPECT_EQ(1, node.vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0}, storage_.local_volume_ids[node.vol_ids[0]]);

    auto const& md = bih_tree.metadata;
    EXPECT_EQ(1, md.num_finite_bboxes);
    EXPECT_EQ(0, md.num_infinite_bboxes);
    EXPECT_EQ(1, md.depth);
}

TEST_F(BIHBuilderTest, multiple_nonpartitionable_volumes)
{
    VecFastBbox bboxes = {
        {{0, 0, 0}, {1, 1, 1}},
        {{0, 0, 0}, {1, 1, 1}},
    };

    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);

    ASSERT_EQ(0, bih_tree.inf_vol_ids.size());
    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());

    auto node = storage_.leaf_nodes[bih_tree.leaf_nodes[0]];
    ASSERT_EQ(BIHNodeId{}, node.parent);
    EXPECT_EQ(2, node.vol_ids.size());
    EXPECT_EQ(LocalVolumeId{0}, storage_.local_volume_ids[node.vol_ids[0]]);
    EXPECT_EQ(LocalVolumeId{1}, storage_.local_volume_ids[node.vol_ids[1]]);

    auto const& md = bih_tree.metadata;
    EXPECT_EQ(2, md.num_finite_bboxes);
    EXPECT_EQ(0, md.num_infinite_bboxes);
    EXPECT_EQ(1, md.depth);
}

TEST_F(BIHBuilderTest, single_infinite_volume)
{
    VecFastBbox bboxes = {FastBBox::from_infinite()};

    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);

    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());
    ASSERT_EQ(1, bih_tree.inf_vol_ids.size());

    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);

    auto const& md = bih_tree.metadata;
    EXPECT_EQ(0, md.num_finite_bboxes);
    EXPECT_EQ(1, md.num_infinite_bboxes);
    EXPECT_EQ(0, md.depth);
}

TEST_F(BIHBuilderTest, multiple_infinite_volumes)
{
    VecFastBbox bboxes = {
        FastBBox::from_infinite(),
        FastBBox::from_infinite(),
    };

    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    auto bih_tree = build(std::move(bboxes), implicit_vol_ids_);

    ASSERT_EQ(0, bih_tree.inner_nodes.size());
    ASSERT_EQ(1, bih_tree.leaf_nodes.size());
    ASSERT_EQ(2, bih_tree.inf_vol_ids.size());

    EXPECT_EQ(LocalVolumeId{0},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[0]]);
    EXPECT_EQ(LocalVolumeId{1},
              storage_.local_volume_ids[bih_tree.inf_vol_ids[1]]);

    auto const& md = bih_tree.metadata;
    EXPECT_EQ(0, md.num_finite_bboxes);
    EXPECT_EQ(2, md.num_infinite_bboxes);
    EXPECT_EQ(0, md.depth);
}

TEST_F(BIHBuilderTest, TEST_IF_CELERITAS_DEBUG(semi_finite_volumes))
{
    VecFastBbox bboxes = {
        {{0, 0, -inff}, {1, 1, inff}},
        {{1, 0, -inff}, {2, 1, inff}},
        {{2, 0, -inff}, {4, 1, inff}},
        {{4, 0, -inff}, {8, 1, inff}},
        {{0, -inff, -inff}, {1, inff, inff}},
        {{-inff, 0, 0}, {inff, 1, 1}},
    };

    BIHBuilder build(&storage_, BIHBuilder::Input{1});
    EXPECT_THROW(build(std::move(bboxes), implicit_vol_ids_), DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
