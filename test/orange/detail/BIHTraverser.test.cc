//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHTraverser.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHTraverser.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "orange/detail/BIHBuilder.hh"
#include "orange/detail/BIHData.hh"
#include "celeritas/Types.hh"

#include "celeritas_test.hh"

using BIHBuilder = celeritas::detail::BIHBuilder;
using BIHInnerNode = celeritas::detail::BIHInnerNode;
using BIHLeafNode = celeritas::detail::BIHLeafNode;
using BIHTraverser = celeritas::detail::BIHTraverser;

namespace celeritas
{
namespace test
{
class BIHTraverserTest : public Test
{
  public:
    void SetUp() {}

  protected:
    std::vector<FastBBox> bboxes_;

    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;

    static constexpr bool valid_volid_(LocalVolumeId vol_id)
    {
        return static_cast<bool>(vol_id);
    };
    static constexpr bool odd_volid_(LocalVolumeId vol_id)
    {
        return vol_id.unchecked_get() % 2 != 0;
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
TEST_F(BIHTraverserTest, basic)
{
    bboxes_.push_back(FastBBox::from_infinite());
    bboxes_.push_back({{0, 0, 0}, {1.6, 1, 100}});
    bboxes_.push_back({{1.2, 0, 0}, {2.8, 1, 100}});
    bboxes_.push_back({{2.8, 0, 0}, {5, 1, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHTraverser traverser(bih_tree, ref_storage_);

    EXPECT_EQ(LocalVolumeId{0}, traverser({0.8, 0.5, 110}, valid_volid_));
    EXPECT_EQ(LocalVolumeId{1}, traverser({0.8, 0.5, 30}, valid_volid_));
    EXPECT_EQ(LocalVolumeId{2}, traverser({2.0, 0.6, 40}, valid_volid_));
    EXPECT_EQ(LocalVolumeId{3}, traverser({2.9, 0.7, 50}, valid_volid_));
    EXPECT_EQ(LocalVolumeId{4}, traverser({2.9, -0.7, 50}, valid_volid_));
    EXPECT_EQ(LocalVolumeId{5}, traverser({2.9, -0.7, 50}, odd_volid_));
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
 */
TEST_F(BIHTraverserTest, grid)
{
    bboxes_.push_back(FastBBox::from_infinite());
    for (auto i : range(3))
    {
        for (auto j : range(4))
        {
            auto x = static_cast<fast_real_type>(i);
            auto y = static_cast<fast_real_type>(j);
            bboxes_.push_back({{x, y, 0}, {x + 1, y + 1, 100}});
        }
    }

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHTraverser traverser(bih_tree, ref_storage_);

    EXPECT_EQ(LocalVolumeId{0}, traverser({0.8, 0.5, 110}, valid_volid_));

    size_type index{1};
    for (auto i : range(3))
    {
        for (auto j : range(4))
        {
            EXPECT_EQ(LocalVolumeId{index++},
                      traverser({0.5 + i, 0.5 + j, 30}, valid_volid_));
        }
    }
}

//---------------------------------------------------------------------------//
// Degenerate, single leaf cases
//---------------------------------------------------------------------------//

TEST_F(BIHTraverserTest, single_finite_volume)
{
    bboxes_.push_back({{0, 0, 0}, {1, 1, 1}});

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHTraverser traverser(bih_tree, ref_storage_);

    EXPECT_EQ(LocalVolumeId{0}, traverser({0.5, 0.5, 0.5}, valid_volid_));
}

TEST_F(BIHTraverserTest, multiple_nonpartitionable_volumes)
{
    bboxes_.push_back({{0, 0, 0}, {1, 1, 1}});
    bboxes_.push_back({{0, 0, 0}, {1, 1, 1}});

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHTraverser traverser(bih_tree, ref_storage_);

    EXPECT_EQ(LocalVolumeId{0}, traverser({0.5, 0.5, 0.5}, valid_volid_));
    EXPECT_EQ(LocalVolumeId{1}, traverser({0.5, 0.5, 0.5}, odd_volid_));
}

TEST_F(BIHTraverserTest, single_infinite_volume)
{
    bboxes_.push_back(FastBBox::from_infinite());

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHTraverser traverser(bih_tree, ref_storage_);

    EXPECT_EQ(LocalVolumeId{0}, traverser({0.5, 0.5, 0.5}, valid_volid_));
}

TEST_F(BIHTraverserTest, multiple_infinite_volumes)
{
    bboxes_.push_back(FastBBox::from_infinite());
    bboxes_.push_back(FastBBox::from_infinite());

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHTraverser traverser(bih_tree, ref_storage_);

    EXPECT_EQ(LocalVolumeId{0}, traverser({0.5, 0.5, 0.5}, valid_volid_));
    EXPECT_EQ(LocalVolumeId{1}, traverser({0.5, 0.5, 0.5}, odd_volid_));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
