//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHIntersectingVolFinder.hh"

#include <limits>
#include <map>

#include "corecel/StringSimplifier.hh"
#include "orange/OrangeParamsOutput.hh"
#include "orange/OrangeTypes.hh"
#include "orange/detail/BIHBuilder.hh"
#include "orange/univ/detail/Types.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

// Mock class with operator() to serve as a visit_vol functor
// Acts as though the volume ID has a single surface with the same ID
class MockIntersector
{
  public:
    using DistMap = std::map<LocalVolumeId, real_type>;

  public:
    explicit MockIntersector(DistMap const& dist_map) : dist_map_(dist_map) {}

    Intersection operator()(LocalVolumeId vol_id, real_type max_distance)
    {
        CELER_EXPECT(vol_id);
        auto iter = dist_map_.find(vol_id);
        if (iter == dist_map_.end())
        {
            ++misses_;
            return {};
        }

        if (iter->second > max_distance)
        {
            // Distance is outside the maximum
            ++misses_;
            return {};
        }

        OnLocalSurface on_surface{LocalSurfaceId{*vol_id}, Sense::outside};
        ++hits_;
        return Intersection{on_surface, iter->second};
    }

    size_type hit_count() const { return hits_; }
    size_type miss_count() const { return misses_; }

  private:
    DistMap const& dist_map_;
    size_type hits_{0};
    size_type misses_{0};
};

struct IntersectResult
{
    static constexpr int no_hit{-1};

    real_type distance{};
    int hit{no_hit};
    std::vector<int> hit_count;
    std::vector<int> miss_count;
};

std::ostream& operator<<(std::ostream& os, IntersectResult const& ref)
{
    // clang-format off
    os << "/*** INTERSECT RESULT ***/\n"
          "IntersectResult ref;\n"
       << CELER_REF_ATTR(distance)
       << CELER_REF_ATTR(hit)
       << CELER_REF_ATTR(hit_count)
       << CELER_REF_ATTR(miss_count)
       << "EXPECT_REF_EQ(ref, result) << result;\n"
          "/*** END CODE ***/\n";
    // clang-format on
    return os;
}

::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   IntersectResult const& val1,
                                   IntersectResult const& val2)
{
    ::celeritas::test::AssertionHelper result{expr1, expr2};

#define IRE_COMPARE(ATTR)                                          \
    if (val1.ATTR != val2.ATTR)                                    \
    {                                                              \
        result.fail() << "Expected " #ATTR ": " << repr(val1.ATTR) \
                      << " but got " << repr(val2.ATTR);           \
    }                                                              \
    else                                                           \
        CELER_DISCARD(int)

    if (!SoftEqual<>{}(val1.distance, val2.distance))
    {
        result.fail() << "Expected distance: " << repr(val1.distance)
                      << " but got " << repr(val2.distance);
    }
    IRE_COMPARE(hit);
    IRE_COMPARE(hit_count);
    IRE_COMPARE(miss_count);

#undef IRE_COMPARE
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Build a BIH tree and test ray intersections with volumes.
 *
 * This class owns the BIH tree storage and provides intersection testing via
 * a locally-constructed \c BIHIntersectingVolFinder.
 */
class LocalBihTreeTester
{
  public:
    using VecBBox = BIHBuilder::VecBBox;
    using Ray = BIHIntersectingVolFinder::Ray;

    LocalBihTreeTester(VecBBox bboxes, BIHBuilder::Input input)
    {
        BIHBuilder build(&storage_, input);
        BIHBuilder::SetLocalVolId implicit_vol_ids;
        bih_tree_ = build(std::move(bboxes), implicit_vol_ids);
        ref_storage_ = storage_;
    }

    template<class F>
    Intersection operator()(Ray ray, F&& visit_vol, real_type max_dist) const
    {
        BIHIntersectingVolFinder find_volume{bih_tree_, ref_storage_};
        return find_volume(ray, std::forward<F>(visit_vol), max_dist);
    }

    friend std::string to_string(LocalBihTreeTester const& btt)
    {
        return dump_bih_structure(btt.bih_tree_, btt.ref_storage_);
    }

  private:
    BIHTreeRecord bih_tree_;
    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;
};

//---------------------------------------------------------------------------//
/* The BIHIntersectingVolFinder class is tested with the following geometry,
 * consisting of partial and fully overlapping bounding boxes.
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
class BIHIntersectingVolFinderTest : public ::celeritas::test::Test
{
  public:
    using Ray = LocalBihTreeTester::Ray;
    using DistMap = MockIntersector::DistMap;
    using VecBBox = BIHBuilder::VecBBox;

  protected:
    void SetUp() override
    {
        testers_.reserve(3);
        for (auto leaf_size : {1, 4, 8})
        {
            inp::BIHBuilder setup;
            setup.max_leaf_size = leaf_size;
            testers_.emplace_back(this->make_bboxes(), setup);
        }
    }

    virtual VecBBox make_bboxes() const = 0;

    // Get results for a ray across all leaf-size intersectors
    IntersectResult
    get_result(Ray ray, DistMap const& dist_map, real_type max_search_dist)
    {
        IntersectResult result;
        for (auto& tester : testers_)
        {
            MockIntersector visit_vol{dist_map};
            auto intersection = tester(ray, visit_vol, max_search_dist);
            auto hit = intersection
                           ? static_cast<int>(intersection.surface.id().value())
                           : IntersectResult::no_hit;
            if (result.hit_count.empty())
            {
                result.distance = intersection.distance;
                result.hit = hit;
            }
            else
            {
                EXPECT_EQ(result.distance, intersection.distance);
                EXPECT_EQ(result.hit, hit);
            }
            result.hit_count.push_back(static_cast<int>(visit_vol.hit_count()));
            result.miss_count.push_back(
                static_cast<int>(visit_vol.miss_count()));
        }
        return result;
    }

    // Get results for a ray across all leaf-size intersectors, with a max
    // search distance
    IntersectResult get_result(Ray ray, DistMap const& dist_map)
    {
        constexpr auto infr = std::numeric_limits<real_type>::infinity();
        return get_result(ray, dist_map, infr);
    }

    std::vector<LocalBihTreeTester> testers_;
};

class PathologicalBihTest : public BIHIntersectingVolFinderTest
{
  public:
    VecBBox make_bboxes() const
    {
        return {
            FastBBox::from_infinite(),
            {{0, 0, 0}, {1.6f, 1, 100}},
            {{1.2f, 0, 0}, {2.8f, 1, 100}},
            {{2.8f, 0, 0}, {5, 1, 100}},
            {{0, -1, 0}, {5, 0, 100}},
            {{0, -1, 0}, {5, 0, 100}},
        };
    }
};

TEST_F(PathologicalBihTest, tree_output)
{
    celeritas::test::StringSimplifier simplify{3};
    ASSERT_EQ(3, testers_.size());
    EXPECT_JSON_EQ(
        R"json({"inf_vol_ids":[0],"tree":[["i","x",[1,2],[2.80,0.0]],["i","x",[3,4],[1.60,1.20]],["i","x",[5,6],[5.0,2.80]],["l",[1]],["l",[2]],["l",[4,5]],["l",[3]]]})json",
        simplify(to_string(testers_[0])));
    EXPECT_JSON_EQ(
        R"json({"inf_vol_ids":[0],"tree":[["i","x",[1,2],[2.80,0.0]],["l",[1,2]],["l",[3,4,5]]]})json",
        simplify(to_string(testers_[1])));
    EXPECT_JSON_EQ(R"json({"inf_vol_ids":[0],"tree":[["l",[1,2,3,4,5]]]})json",
                   simplify(to_string(testers_[2])));
}

// Test the case where the ray starts outside the bbox and the first bbox
// intersection yields the first volume intersection.
TEST_F(PathologicalBihTest, outside_first)
{
    Real3 pos, dir;
    DistMap dist_map;

    // Ray intersects V1 from the left
    pos = {-1., 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {
        {LocalVolumeId{0}, 10},
        {LocalVolumeId{1}, 1},
        {LocalVolumeId{2}, 1.2},
        {LocalVolumeId{3}, 2.8},
    };
    {
        IntersectResult ref;
        ref.distance = 1.0;
        ref.hit = 1;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V2 from above
    pos = {2., 2., 50.};
    dir = {0., -1., 0.};
    dist_map = {
        {LocalVolumeId{2}, 1.},
        {LocalVolumeId{4}, 2.},
        {LocalVolumeId{5}, 2.},
    };
    {
        IntersectResult ref;
        ref.distance = 1;
        ref.hit = 2;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V3 from the right
    pos = {6, 0.5, 50.};
    dir = {-1., 0., 0.};
    dist_map = {
        {LocalVolumeId{1}, 4.4},
        {LocalVolumeId{2}, 3.2},
        {LocalVolumeId{3}, 1.},
    };
    {
        IntersectResult ref;
        ref.distance = 1.0;
        ref.hit = 3;
        ref.hit_count = {3, 3, 3};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V4 from the left
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.2}, {LocalVolumeId{5}, 1.3}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.hit = 4;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {2, 2, 2};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.hit = 5;
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left, max search distance is closer
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 1.1;
        ref.hit_count = {0, 0, 0};
        ref.miss_count = {3, 3, 3};
        auto result = this->get_result({pos, dir}, dist_map, 1.1);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left, max search distance is further
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.hit = 5;
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 1.3);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

// Test the case where the ray starts somewhere inside a bbox and this bbox
// contains first intersecting volume.
TEST_F(PathologicalBihTest, inside_first)
{
    Real3 pos, dir;
    DistMap dist_map;

    // Ray starts in VO and intersects V0
    pos = {-1., 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {
        {LocalVolumeId{0}, 0.5},
        {LocalVolumeId{1}, 1},
        {LocalVolumeId{2}, 1.2},
        {LocalVolumeId{3}, 2.8},
    };
    {
        IntersectResult ref;
        ref.distance = 0.5;
        ref.hit = 0;
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {0, 0, 0};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray starts in V1 and intersects V1
    pos = {1., 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {
        {LocalVolumeId{0}, 10},
        {LocalVolumeId{1}, 0.1},
        {LocalVolumeId{2}, 0.7},
        {LocalVolumeId{3}, 2.3},
    };
    {
        IntersectResult ref;
        ref.distance = 0.1;
        ref.hit = 1;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray starts in V2 and intersects V2
    pos = {2., 2., 50.};
    dir = {0., -1., 0.};
    dist_map = {
        {LocalVolumeId{2}, 1.}, {LocalVolumeId{4}, 2.}, {LocalVolumeId{5}, 2.}};
    {
        IntersectResult ref;
        ref.distance = 1;
        ref.hit = 2;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray starts in V3 and intersects V3
    pos = {4, 0.5, 50.};
    dir = {-1., 0., 0.};
    dist_map = {
        {LocalVolumeId{1}, 2.4},
        {LocalVolumeId{2}, 1.2},
        {LocalVolumeId{3}, 1.},
    };
    {
        IntersectResult ref;
        ref.distance = 1.0;
        ref.hit = 3;
        ref.hit_count = {3, 3, 3};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V4 from the left
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.2}, {LocalVolumeId{5}, 1.3}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.hit = 4;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {2, 2, 2};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.hit = 5;
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left, max search distance is closer
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 0.1;
        ref.hit_count = {0, 0, 0};
        ref.miss_count = {3, 3, 3};
        auto result = this->get_result({pos, dir}, dist_map, 0.1);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left, max search distance is further
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.hit = 5;
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 1.6);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

// Test the case where the first intersection does not yields the first volume
// collision
TEST_F(PathologicalBihTest, not_first)
{
    Real3 pos, dir;
    DistMap dist_map;

    // Ray goes through V1 but intersects with V2 first
    pos = {-0.5, 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {
        {LocalVolumeId{1}, 2.0},
        {LocalVolumeId{2}, 1.7},
        {LocalVolumeId{3}, 3.3},
    };
    {
        IntersectResult ref;
        ref.distance = 1.7;
        ref.hit = 2;
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray goes all the way through V1, V2 and V3, intersects V0
    pos = {-0.5, 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, 11.}};
    {
        IntersectResult ref;
        ref.distance = 11.;
        ref.hit = 0;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {3, 3, 3};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray goes through V4 and V5 and intersects with V2
    pos = {1.5, -2, 50.};
    dir = {0., 1., 0.};
    dist_map = {{LocalVolumeId{2}, 1.5}};
    {
        IntersectResult ref;
        ref.distance = 1.5;
        ref.hit = 2;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {4, 4, 4};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray goes through V4 and V5 and intersects with V2, max search is
    // closer
    pos = {1.5, -2, 50.};
    dir = {0., 1., 0.};
    dist_map = {{LocalVolumeId{2}, 1.5}};
    {
        IntersectResult ref;
        ref.distance = 0.8;
        ref.hit_count = {0, 0, 0};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 0.8);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray goes through V4 and V5 and intersects with V2, max search is
    // further
    pos = {1.5, -2, 50.};
    dir = {0., 1., 0.};
    dist_map = {{LocalVolumeId{2}, 1.5}};
    {
        IntersectResult ref;
        ref.distance = 1.5;
        ref.hit = 2;
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {4, 4, 4};
        auto result = this->get_result({pos, dir}, dist_map, 2.1);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
