//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BvhEnclosingVolFinder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BvhEnclosingVolFinder.hh"

#include "orange/detail/BvhBuilder.hh"
#include "orange/detail/BvhData.hh"

#include "celeritas_test.hh"

using BvhBuilder = celeritas::detail::BvhBuilder;
using BvhInternalNode = celeritas::detail::BvhInternalNode;
using BvhLeafNode = celeritas::detail::BvhLeafNode;
using BvhEnclosingVolFinder = celeritas::detail::BvhEnclosingVolFinder;

namespace celeritas
{
namespace test
{
class BvhEnclosingVolFinderTest : public Test
{
  public:
    using VecFastBbox = BvhBuilder::VecBBox;

  protected:
    //! Accept a volume if the point is inside its bbox
    struct IsPointInsideBboxVolume
    {
        VecFastBbox const& bboxes;
        Real3 pos;

        bool operator()(LocalVolumeId vol_id) const
        {
            CELER_EXPECT(vol_id < bboxes.size());
            return is_inside(bboxes[*vol_id], pos);
        }
    };

    //! Accept a volume if the point is inside its bbox, and vol_id is odd
    struct IsPointInsideOddBboxVolume
    {
        VecFastBbox const& bboxes;
        Real3 pos;

        bool operator()(LocalVolumeId vol_id) const
        {
            CELER_EXPECT(vol_id < bboxes.size());
            return vol_id.unchecked_get() % 2 != 0
                   && is_inside(bboxes[*vol_id], pos);
        }
    };

  protected:
    HostVal<detail::BvhTreeData> storage_;
    HostCRef<detail::BvhTreeData> ref_storage_;
    BvhBuilder::SetLocalVolId implicit_vol_ids_;
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
 */
TEST_F(BvhEnclosingVolFinderTest, basic)
{
    auto run_test = [&](size_type max_leaf_size) {
        VecFastBbox bboxes = {
            FastBBox::from_infinite(),
            {{0, 0, 0}, {1.6f, 1, 100}},
            {{1.2f, 0, 0}, {2.8f, 1, 100}},
            {{2.8f, 0, 0}, {5, 1, 100}},
            {{0, -1, 0}, {5, 0, 100}},
            {{0, -1, 0}, {5, 0, 100}},
        };

        BvhBuilder build(&storage_, BvhBuilder::Input{max_leaf_size});
        auto bvh_tree = build(VecFastBbox{bboxes}, implicit_vol_ids_);

        ref_storage_ = storage_;
        BvhEnclosingVolFinder find_volume(bvh_tree, ref_storage_);

        Real3 pos = {0.8, 0.5, 110};
        EXPECT_EQ(LocalVolumeId{0},
                  find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));

        pos = {0.8, 0.5, 30};
        EXPECT_EQ(LocalVolumeId{1},
                  find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));

        pos = {2.0, 0.6, 40};
        EXPECT_EQ(LocalVolumeId{2},
                  find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));

        pos = {2.9, 0.7, 50};
        EXPECT_EQ(LocalVolumeId{3},
                  find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));

        pos = {2.9, -0.7, 50};
        EXPECT_EQ(LocalVolumeId{4},
                  find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));

        pos = {2.9, -0.7, 50};
        EXPECT_EQ(LocalVolumeId{5},
                  find_volume(pos, IsPointInsideOddBboxVolume{bboxes, pos}));
    };

    for (auto max_leaf_size : range(1, 4))
    {
        run_test(max_leaf_size);
    }
}

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
TEST_F(BvhEnclosingVolFinderTest, grid)
{
    auto run_test = [&](size_type max_leaf_size) {
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

        BvhBuilder build(&storage_, BvhBuilder::Input{max_leaf_size});
        auto bvh_tree = build(VecFastBbox{bboxes}, implicit_vol_ids_);

        ref_storage_ = storage_;
        BvhEnclosingVolFinder find_volume(bvh_tree, ref_storage_);

        Real3 outside_pos{0.8, 0.5, 110};
        EXPECT_EQ(LocalVolumeId{0},
                  find_volume(outside_pos,
                              IsPointInsideBboxVolume{bboxes, outside_pos}));

        size_type index{1};
        for (auto i : range(3))
        {
            for (auto j : range(4))
            {
                constexpr real_type half{0.5};
                Real3 pos{half + i, half + j, 30};
                EXPECT_EQ(
                    LocalVolumeId{index++},
                    find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));
            }
        }
    };

    for (auto max_leaf_size : range(1, 4))
    {
        run_test(max_leaf_size);
    }
}

//---------------------------------------------------------------------------//
// Degenerate, single leaf cases
//---------------------------------------------------------------------------//

TEST_F(BvhEnclosingVolFinderTest, single_finite_volume)
{
    VecFastBbox bboxes = {{{0, 0, 0}, {1, 1, 1}}};

    BvhBuilder build(&storage_, BvhBuilder::Input{});
    auto bvh_tree = build(VecFastBbox{bboxes}, implicit_vol_ids_);

    ref_storage_ = storage_;
    BvhEnclosingVolFinder find_volume(bvh_tree, ref_storage_);

    Real3 pos{0.5, 0.5, 0.5};
    EXPECT_EQ(LocalVolumeId{0},
              find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));
}

TEST_F(BvhEnclosingVolFinderTest, multiple_nonpartitionable_volumes)
{
    VecFastBbox bboxes = {
        {{0, 0, 0}, {1, 1, 1}},
        {{0, 0, 0}, {1, 1, 1}},
    };

    BvhBuilder build(&storage_, BvhBuilder::Input{});
    auto bvh_tree = build(VecFastBbox{bboxes}, implicit_vol_ids_);

    ref_storage_ = storage_;
    BvhEnclosingVolFinder find_volume(bvh_tree, ref_storage_);

    Real3 pos{0.5, 0.5, 0.5};
    EXPECT_EQ(LocalVolumeId{0},
              find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));
    EXPECT_EQ(LocalVolumeId{1},
              find_volume(pos, IsPointInsideOddBboxVolume{bboxes, pos}));
}

TEST_F(BvhEnclosingVolFinderTest, single_infinite_volume)
{
    VecFastBbox bboxes = {FastBBox::from_infinite()};

    BvhBuilder build(&storage_, BvhBuilder::Input{});
    auto bvh_tree = build(VecFastBbox{bboxes}, implicit_vol_ids_);

    ref_storage_ = storage_;
    BvhEnclosingVolFinder find_volume(bvh_tree, ref_storage_);

    Real3 pos{0.5, 0.5, 0.5};
    EXPECT_EQ(LocalVolumeId{0},
              find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));
}

TEST_F(BvhEnclosingVolFinderTest, multiple_infinite_volumes)
{
    VecFastBbox bboxes = {
        FastBBox::from_infinite(),
        FastBBox::from_infinite(),
    };

    BvhBuilder build(&storage_, BvhBuilder::Input{});
    auto bvh_tree = build(VecFastBbox{bboxes}, implicit_vol_ids_);

    ref_storage_ = storage_;
    BvhEnclosingVolFinder find_volume(bvh_tree, ref_storage_);

    Real3 pos{0.5, 0.5, 0.5};
    EXPECT_EQ(LocalVolumeId{0},
              find_volume(pos, IsPointInsideBboxVolume{bboxes, pos}));
    EXPECT_EQ(LocalVolumeId{1},
              find_volume(pos, IsPointInsideOddBboxVolume{bboxes, pos}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
