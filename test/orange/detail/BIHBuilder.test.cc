//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHBuilder.hh"

#include <limits>
#include <vector>

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/Types.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "geocel/Types.hh"
#include "orange/detail/BIHData.hh"
#include "orange/detail/BIHView.hh"

#include "celeritas_test.hh"

using celeritas::test::id_to_int;

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
    using Input = BIHBuilder::Input;
    using VecInt = std::vector<int>;
    using Side = BIHInternalNode::Side;
    using Real3 = FastBBox::Real3;

  protected:
    static constexpr auto inff
        = std::numeric_limits<fast_real_type>::infinity();

    //! Build BIH tree data with a single tree in it and one volume per leaf
    void build(VecFastBbox bboxes)
    {
        Input input;
        input.max_leaf_size = 1;
        return this->build(std::move(bboxes), std::move(input));
    }

    //! Build BIH tree data with a single tree in it and one volume per leaf
    void build(VecFastBbox&& bboxes, Input&& input)
    {
        HostVal<BIHTreeData> data;
        BIHBuilder build(&data, std::move(input));
        tree_ = build(std::move(bboxes), {});
        store_ = ParamsDataStore<BIHTreeData>{std::move(data)};
        ASSERT_TRUE(tree_);
        ASSERT_TRUE(store_);
    }

    ParamsDataStore<BIHTreeData> store_;
    BIHTreeRecord tree_;
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
    this->build({
        FastBBox::from_infinite(),
        {{0, 0, 0}, {1.6f, 1, 100}},
        {{1.2f, 0, 0}, {2.8f, 1, 100}},
        {{2.8f, 0, 0}, {5, 1, 100}},
        {{0, -1, 0}, {5, 0, 100}},
        {{0, -1, 0}, {5, 0, 100}},
    });

    // Test nodes
    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(3, view.num_internal_nodes());
    ASSERT_EQ(4, view.num_leaf_nodes());
    EXPECT_VEC_EQ(VecInt({0}), id_to_int(view.inf_vol_ids()));

    // N0, I0
    {
        auto const& node = view.inner_node(BIHNodeId{0});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(2.8f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(0.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{1}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{2}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.8f, 1.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, -1.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({5.f, 1.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N1, I1
    {
        auto const& node = view.inner_node(BIHNodeId{1});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(1.6f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.2f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{3}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{4}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.6f, 1.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.2f, 0.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.8f, 1.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N2, I2
    {
        auto const& node = view.inner_node(BIHNodeId{2});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(5.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(2.8f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{5}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{6}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, -1.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({5.f, 0.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.8f, 0.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({5.f, 1.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    EXPECT_VEC_EQ(VecInt({1}), id_to_int(view.leaf_vol_ids(BIHNodeId{3})));
    EXPECT_VEC_EQ(VecInt({2}), id_to_int(view.leaf_vol_ids(BIHNodeId{4})));
    EXPECT_VEC_EQ(VecInt({4, 5}), id_to_int(view.leaf_vol_ids(BIHNodeId{5})));
    EXPECT_VEC_EQ(VecInt({3}), id_to_int(view.leaf_vol_ids(BIHNodeId{6})));

    // Metadata
    {
        auto const& md = tree_.metadata;
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
    void build_grid(Input&& input)
    {
        VecFastBbox bboxes = {FastBBox::from_infinite()};
        for (auto i : range(3))
        {
            for (auto j : range(4))
            {
                auto x = static_cast<fast_real_type>(i);
                auto y = static_cast<fast_real_type>(j);
                bboxes.push_back({{x, y, 0}, {x + 1, y + 1, 100}});
            }
        }
        this->build(std::move(bboxes), std::move(input));
    }
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
    this->build_grid([] {
        Input i;
        i.max_leaf_size = 1;
        return i;
    }());

    // Test nodes
    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(11, view.num_internal_nodes());
    ASSERT_EQ(12, view.num_leaf_nodes());
    EXPECT_VEC_EQ(VecInt({0}), id_to_int(view.inf_vol_ids()));

    // N0, I0
    {
        auto const& node = view.inner_node(BIHNodeId{0});

        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{1}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{6}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 2.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 4.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N1, I1
    {
        auto const& node = view.inner_node(BIHNodeId{1});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{2}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{3}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 0.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N2, I2
    {
        auto const& node = view.inner_node(BIHNodeId{2});

        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{11}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{12}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 1.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 1.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N3, I3
    {
        auto const& node = view.inner_node(BIHNodeId{3});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{4}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{5}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 0.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N4, I4
    {
        auto const& node = view.inner_node(BIHNodeId{4});

        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{13}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{14}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 1.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 1.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N5, I5
    {
        auto const& node = view.inner_node(BIHNodeId{5});

        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{15}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{16}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 1.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 1.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    EXPECT_VEC_EQ(VecInt({1}), id_to_int(view.leaf_vol_ids(BIHNodeId{11})));
    EXPECT_VEC_EQ(VecInt({2}), id_to_int(view.leaf_vol_ids(BIHNodeId{12})));
    EXPECT_VEC_EQ(VecInt({5}), id_to_int(view.leaf_vol_ids(BIHNodeId{13})));
    EXPECT_VEC_EQ(VecInt({6}), id_to_int(view.leaf_vol_ids(BIHNodeId{14})));
    EXPECT_VEC_EQ(VecInt({9}), id_to_int(view.leaf_vol_ids(BIHNodeId{15})));
    EXPECT_VEC_EQ(VecInt({10}), id_to_int(view.leaf_vol_ids(BIHNodeId{16})));

    // Metadata
    {
        auto const& md = tree_.metadata;
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
    this->build_grid([] {
        Input i;
        i.max_leaf_size = 4;
        return i;
    }());

    // Test nodes
    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(3, view.num_internal_nodes());
    ASSERT_EQ(4, view.num_leaf_nodes());
    EXPECT_VEC_EQ(VecInt({0}), id_to_int(view.inf_vol_ids()));

    // N0, I0
    {
        auto const& node = view.inner_node(BIHNodeId{0});

        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{1}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{2}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 2.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 4.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N1, I1
    {
        auto const& node = view.inner_node(BIHNodeId{1});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{3}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{4}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 0.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N2, I2
    {
        auto const& node = view.inner_node(BIHNodeId{2});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{5}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{6}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 2.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 4.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 4.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    EXPECT_VEC_EQ(VecInt({1, 2}), id_to_int(view.leaf_vol_ids(BIHNodeId{3})));
    EXPECT_VEC_EQ(VecInt({5, 6, 9, 10}),
                  id_to_int(view.leaf_vol_ids(BIHNodeId{4})));
    EXPECT_VEC_EQ(VecInt({3, 4}), id_to_int(view.leaf_vol_ids(BIHNodeId{5})));
    EXPECT_VEC_EQ(VecInt({7, 8, 11, 12}),
                  id_to_int(view.leaf_vol_ids(BIHNodeId{6})));

    // Metadata
    {
        auto const& md = tree_.metadata;
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
    this->build_grid([] {
        Input i;
        i.max_leaf_size = 1;
        i.depth_limit = 4;
        return i;
    }());

    // Test nodes
    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(7, view.num_internal_nodes());
    ASSERT_EQ(8, view.num_leaf_nodes());
    EXPECT_VEC_EQ(VecInt({0}), id_to_int(view.inf_vol_ids()));

    // N0, I0
    {
        auto const& node = view.inner_node(BIHNodeId{0});

        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{1}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{4}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 2.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 4.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N1, I1
    {
        auto const& node = view.inner_node(BIHNodeId{1});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{2}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{3}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 0.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N2, I2
    {
        auto const& node = view.inner_node(BIHNodeId{2});

        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_EQ(Axis{1}, node.axis());
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(1.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{7}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{8}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 1.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({0.f, 1.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    // N3, I3
    {
        auto const& node = view.inner_node(BIHNodeId{3});

        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_EQ(Axis{0}, node.axis());
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::left));
        EXPECT_SOFT_EQ(2.f, node.bounding_plane_pos(Side::right));
        EXPECT_EQ(BIHNodeId{9}, node.child(Side::left));
        EXPECT_EQ(BIHNodeId{10}, node.child(Side::right));

        EXPECT_VEC_SOFT_EQ(VecFastReal({1.f, 0.f, 0.f}),
                           node.bbox(Side::left).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 2.f, 100.f}),
                           node.bbox(Side::left).upper());
        EXPECT_VEC_SOFT_EQ(VecFastReal({2.f, 0.f, 0.f}),
                           node.bbox(Side::right).lower());
        EXPECT_VEC_SOFT_EQ(VecFastReal({3.f, 2.f, 100.f}),
                           node.bbox(Side::right).upper());
    }

    EXPECT_VEC_EQ(VecInt({1}), id_to_int(view.leaf_vol_ids(BIHNodeId{7})));
    EXPECT_VEC_EQ(VecInt({2}), id_to_int(view.leaf_vol_ids(BIHNodeId{8})));
    EXPECT_VEC_EQ(VecInt({5, 6}), id_to_int(view.leaf_vol_ids(BIHNodeId{9})));
    EXPECT_VEC_EQ(VecInt({9, 10}), id_to_int(view.leaf_vol_ids(BIHNodeId{10})));

    // Metadata
    {
        auto const& md = tree_.metadata;
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
    this->build({{{0, 0, 0}, {1, 1, 1}}});

    ASSERT_EQ(0, tree_.inf_vol_ids.size());
    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(0, view.num_internal_nodes());
    ASSERT_EQ(1, view.num_leaf_nodes());

    EXPECT_VEC_EQ(VecInt({0}), id_to_int(view.leaf_vol_ids(BIHNodeId{0})));

    auto const& md = tree_.metadata;
    EXPECT_EQ(1, md.num_finite_bboxes);
    EXPECT_EQ(0, md.num_infinite_bboxes);
    EXPECT_EQ(1, md.depth);
}

TEST_F(BIHBuilderTest, multiple_nonpartitionable_volumes)
{
    this->build({{{0, 0, 0}, {1, 1, 1}}, {{0, 0, 0}, {1, 1, 1}}});

    ASSERT_EQ(0, tree_.inf_vol_ids.size());
    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(0, view.num_internal_nodes());
    ASSERT_EQ(1, view.num_leaf_nodes());

    EXPECT_VEC_EQ(VecInt({0, 1}), id_to_int(view.leaf_vol_ids(BIHNodeId{0})));

    auto const& md = tree_.metadata;
    EXPECT_EQ(2, md.num_finite_bboxes);
    EXPECT_EQ(0, md.num_infinite_bboxes);
    EXPECT_EQ(1, md.depth);
}

TEST_F(BIHBuilderTest, single_infinite_volume)
{
    this->build({FastBBox::from_infinite()});

    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(0, view.num_internal_nodes());
    ASSERT_EQ(1, view.num_leaf_nodes());
    EXPECT_VEC_EQ(VecInt({0}), id_to_int(view.inf_vol_ids()));

    auto const& md = tree_.metadata;
    EXPECT_EQ(0, md.num_finite_bboxes);
    EXPECT_EQ(1, md.num_infinite_bboxes);
    EXPECT_EQ(0, md.depth);
}

TEST_F(BIHBuilderTest, multiple_infinite_volumes)
{
    this->build({FastBBox::from_infinite(), FastBBox::from_infinite()});

    BIHView view{tree_, store_.host_ref()};
    ASSERT_EQ(0, view.num_internal_nodes());
    ASSERT_EQ(1, view.num_leaf_nodes());
    EXPECT_VEC_EQ(VecInt({0, 1}), id_to_int(view.inf_vol_ids()));

    auto const& md = tree_.metadata;
    EXPECT_EQ(0, md.num_finite_bboxes);
    EXPECT_EQ(2, md.num_infinite_bboxes);
    EXPECT_EQ(0, md.depth);
}

TEST_F(BIHBuilderTest, TEST_IF_CELERITAS_DEBUG(semi_finite_volumes))
{
    EXPECT_THROW(this->build({
                     {{0, 0, -inff}, {1, 1, inff}},
                     {{1, 0, -inff}, {2, 1, inff}},
                     {{2, 0, -inff}, {4, 1, inff}},
                     {{4, 0, -inff}, {8, 1, inff}},
                     {{0, -inff, -inff}, {1, inff, inff}},
                     {{-inff, 0, 0}, {inff, 1, 1}},
                 }),
                 DebugError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
