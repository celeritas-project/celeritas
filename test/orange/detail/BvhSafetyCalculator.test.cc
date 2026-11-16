//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BvhSafetyCalculator.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BvhSafetyCalculator.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "orange/OrangeTypes.hh"
#include "orange/detail/BvhBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Build a BVH tree and calculate safeties with mock per-volume bounds.
 *
 * The functor given to the calculator returns the bound assigned to a volume
 * (zero if unassigned) and records which volumes were evaluated.
 */
class LocalBvhSafetyTester
{
  public:
    using VecBBox = BvhBuilder::VecBBox;
    using BoundMap = std::unordered_map<LocalVolumeId, real_type>;

    struct Result
    {
        real_type safety{};
        std::vector<int> visited;  //!< Sorted IDs of evaluated volumes
    };

    LocalBvhSafetyTester(VecBBox bboxes, size_type max_leaf_size)
    {
        inp::BvhBuilder input;
        input.max_leaf_size = max_leaf_size;
        BvhBuilder build(&storage_, input);
        bvh_tree_ = build(std::move(bboxes), BvhBuilder::SetLocalVolId{});
        ref_storage_ = storage_;
    }

    Result operator()(
        Real3 const& pos, BoundMap const& bounds, real_type max_dist) const
    {
        Result result;
        auto calc_vol_safety = [&](LocalVolumeId vol_id) -> real_type {
            result.visited.push_back(static_cast<int>(vol_id.get()));
            auto iter = bounds.find(vol_id);
            return iter != bounds.end() ? iter->second : 0;
        };

        BvhSafetyCalculator calc_safety{bvh_tree_, ref_storage_};
        result.safety = calc_safety(pos, calc_vol_safety, max_dist);
        std::sort(result.visited.begin(), result.visited.end());
        return result;
    }

  private:
    BvhTreeRecord bvh_tree_;
    BvhTreeData<Ownership::value, MemSpace::host> storage_;
    BvhTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;
};

//---------------------------------------------------------------------------//

class BvhSafetyCalculatorTest : public ::celeritas::test::Test
{
  protected:
    using VecBBox = LocalBvhSafetyTester::VecBBox;
    using BoundMap = LocalBvhSafetyTester::BoundMap;

    static constexpr auto large = 10000_r;
};

//---------------------------------------------------------------------------//
/* Partial and fully overlapping bounding boxes, with one infinite volume (V0).
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
TEST_F(BvhSafetyCalculatorTest, basic)
{
    VecBBox const bboxes = {
        FastBBox::from_infinite(),
        {{0, 0, 0}, {1.6f, 1, 100}},
        {{1.2f, 0, 0}, {2.8f, 1, 100}},
        {{2.8f, 0, 0}, {5, 1, 100}},
        {{0, -1, 0}, {5, 0, 100}},
        {{0, -1, 0}, {5, 0, 100}},
    };

    // The infinite volume is always evaluated: keep it out of the way
    BoundMap const far_inf = {{LocalVolumeId{0}, large}};
    static int const only_inf[] = {0};

    for (size_type max_leaf_size : {1, 2, 4, 8})
    {
        SCOPED_TRACE("max_leaf_size=" + std::to_string(max_leaf_size));
        LocalBvhSafetyTester calc_safety(bboxes, max_leaf_size);

        {
            SCOPED_TRACE("left of V1");
            auto result = calc_safety({-2, 0.5, 50}, far_inf, large);
            EXPECT_SOFT_EQ(2.0, result.safety);
            // V2 and V3 are skipped because V1 is nearer and visited first
            EXPECT_EQ(
                0, std::count(result.visited.begin(), result.visited.end(), 2));
            EXPECT_EQ(
                0, std::count(result.visited.begin(), result.visited.end(), 3));
        }
        {
            SCOPED_TRACE("volume bound is tighter than its bbox");
            auto bounds = far_inf;
            bounds[LocalVolumeId{1}] = 3.0;
            auto result = calc_safety({-2, 0.5, 50}, bounds, large);
            // V4/V5 are diagonally below: (2, 0.5) away
            EXPECT_SOFT_EQ(std::sqrt(4.25), result.safety);
        }
        {
            SCOPED_TRACE("nothing within the maximum distance");
            auto result = calc_safety({-2, 0.5, 50}, far_inf, 1.5);
            EXPECT_SOFT_EQ(1.5, result.safety);
            EXPECT_VEC_EQ(only_inf, result.visited);
        }
        {
            SCOPED_TRACE("inside overlapping bboxes");
            auto bounds = far_inf;
            bounds[LocalVolumeId{1}] = 0.3;
            bounds[LocalVolumeId{2}] = 0.1;
            auto result = calc_safety({1.4, 0.5, 50}, bounds, large);
            EXPECT_SOFT_EQ(0.1, result.safety);
            static int const expected_visited[] = {0, 1, 2};
            EXPECT_VEC_EQ(expected_visited, result.visited);
        }
        {
            SCOPED_TRACE("inside a bbox without a volume bound");
            auto result = calc_safety({2, -0.5, 50}, far_inf, large);
            EXPECT_EQ(0, result.safety);
            // The search stops at the first of V4 and V5
            EXPECT_EQ(
                1,
                std::count(result.visited.begin(), result.visited.end(), 4)
                    + std::count(
                        result.visited.begin(), result.visited.end(), 5));
        }
        {
            SCOPED_TRACE("infinite volume is nearest");
            BoundMap const bounds = {{LocalVolumeId{0}, 0.5}};
            auto result = calc_safety({-2, 0.5, 50}, bounds, large);
            EXPECT_SOFT_EQ(0.5, result.safety);
            EXPECT_VEC_EQ(only_inf, result.visited);
        }
        {
            SCOPED_TRACE("infinite volume has no bound");
            auto result = calc_safety({-2, 0.5, 50}, {}, large);
            EXPECT_EQ(0, result.safety);
            EXPECT_VEC_EQ(only_inf, result.visited);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Test a large number of adjacent bounding boxes along the Z axis.
 *
 * The extents of the problem are {-1, -1, 0}, {1, 1, num_boxes}.
 */
TEST_F(BvhSafetyCalculatorTest, kebab)
{
    using FastReal3 = FastBBox::Real3;

    constexpr size_type num_boxes{1024};
    VecBBox bboxes;
    bboxes.reserve(num_boxes);
    for (auto i : range(num_boxes))
    {
        bboxes.emplace_back(
            FastReal3{-1, -1, static_cast<fast_real_type>(i)},
            FastReal3{1, 1, static_cast<fast_real_type>(i + 1)});
    }

    for (size_type max_leaf_size : {1, 4, 16})
    {
        SCOPED_TRACE("max_leaf_size=" + std::to_string(max_leaf_size));
        LocalBvhSafetyTester calc_safety(bboxes, max_leaf_size);

        {
            SCOPED_TRACE("beside the middle");
            auto result = calc_safety({3, 0, 500.5}, {}, large);
            EXPECT_SOFT_EQ(2.0, result.safety);
            // Only volumes sharing a leaf with the nearest can be evaluated
            EXPECT_GE(max_leaf_size, result.visited.size());
            EXPECT_EQ(
                1,
                std::count(result.visited.begin(), result.visited.end(), 500));
        }
        {
            SCOPED_TRACE("beyond the end");
            auto result = calc_safety({0, 0, -5.5}, {}, large);
            EXPECT_SOFT_EQ(5.5, result.safety);
            static int const expected_visited[] = {0};
            EXPECT_VEC_EQ(expected_visited, result.visited);
        }
        {
            SCOPED_TRACE("nothing within the maximum distance");
            auto result = calc_safety({3, 0, 500.5}, {}, 1.0);
            EXPECT_SOFT_EQ(1.0, result.safety);
            EXPECT_TRUE(result.visited.empty());
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
