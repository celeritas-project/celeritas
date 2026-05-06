//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHIntersectingVolFinder.hh"

#include <limits>
#include <unordered_map>
#include <vector>

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
    using DistMap = std::unordered_map<LocalVolumeId, real_type>;

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
    real_type distance{};
    LocalSurfaceId intersect_surface;

    /*
     * These vectors, with each element corresponding to the BIH tree
     * tested (configured with different leaf size counts), are diagnostics
     * collected by the MockIntersector class: the number of *volume*
     * intersection tests (after the bbox was used to exclude intersections)
     * that result in a hit, and the number that result in a miss.
     * Their sum is the number of total "distance to intersection" calls.
     */
    std::vector<int> hit_count;
    std::vector<int> miss_count;
};

std::ostream& operator<<(std::ostream& os, IntersectResult const& ref)
{
    // clang-format off
    os << "/*** INTERSECT RESULT ***/\n"
          "IntersectResult ref;\n"
       << CELER_REF_ATTR(distance)
       << CELER_REF_ATTR(intersect_surface)
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

    if (!testdetail::make_soft_comparator<real_type>()(val1.distance,
                                                       val2.distance))
    {
        result.fail() << "Expected distance: " << repr(val1.distance)
                      << " but got " << repr(val2.distance);
    }
    IRE_COMPARE(intersect_surface);
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
    using VecSetup = std::vector<inp::BIHBuilder>;

    static constexpr auto infr = std::numeric_limits<real_type>::infinity();

  protected:
    void SetUp() override
    {
        auto bboxes = this->make_bboxes();
        for (auto&& setup : this->make_bih_setups())
        {
            testers_.emplace_back(bboxes, std::move(setup));
        }
    }

    //! Specify the BIH construction parameters for each tester
    virtual VecSetup make_bih_setups() const = 0;

    //! Specify the bounding box construction
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
            if (result.hit_count.empty())
            {
                result.distance = intersection.distance;
                result.intersect_surface = intersection.surface.id();
            }
            else
            {
                EXPECT_EQ(result.distance, intersection.distance);
                EXPECT_EQ(result.intersect_surface, intersection.surface.id());
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
        return get_result(ray, dist_map, infr);
    }

    auto get_bih_json_strings() const
    {
        std::vector<std::string> result;
        celeritas::test::StringSimplifier simplify{3};
        for (auto& t : testers_)
        {
            result.push_back(simplify(to_string(t)));
        }
        return result;
    }

  private:
    std::vector<LocalBihTreeTester> testers_;
};

class BasicBihTest : public BIHIntersectingVolFinderTest
{
  public:
    VecSetup make_bih_setups() const override
    {
        VecSetup result;
        for (auto leaf_size : {1, 4, 8})
        {
            inp::BIHBuilder setup;
            setup.max_leaf_size = leaf_size;
            result.push_back(setup);
        }
        return result;
    }

    VecBBox make_bboxes() const override
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

TEST_F(BasicBihTest, tree_output)
{
    auto trees = this->get_bih_json_strings();
    ASSERT_EQ(3, trees.size());
    EXPECT_JSON_EQ(
        R"json({"inf_vol_ids":[0],"tree":[["i","x",[1,2],[2.80,0.0]],["i","x",[3,4],[1.60,1.20]],["i","x",[5,6],[5.0,2.80]],["l",[1]],["l",[2]],["l",[4,5]],["l",[3]]]})json",
        trees[0]);
    EXPECT_JSON_EQ(
        R"json({"inf_vol_ids":[0],"tree":[["i","x",[1,2],[2.80,0.0]],["l",[1,2]],["l",[3,4,5]]]})json",
        trees[1]);
    EXPECT_JSON_EQ(R"json({"inf_vol_ids":[0],"tree":[["l",[1,2,3,4,5]]]})json",
                   trees[2]);
}

// Test the case where the ray starts outside the bbox and the first bbox
// intersection yields the first volume intersection.
TEST_F(BasicBihTest, outside_first)
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
        ref.intersect_surface = LocalSurfaceId{1};
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
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
        ref.intersect_surface = LocalSurfaceId{2};
        ref.hit_count = {2, 2, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
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
        ref.intersect_surface = LocalSurfaceId{3};
        ref.hit_count = {1, 1, 3};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V4 from the left
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.2}, {LocalVolumeId{5}, 1.3}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.intersect_surface = LocalSurfaceId{4};
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {2, 2, 2};
        auto result = this->get_result({pos, dir}, dist_map, infr);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.intersect_surface = LocalSurfaceId{5};
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
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
        ref.intersect_surface = LocalSurfaceId{5};
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 1.3);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

// Test the case where the ray starts somewhere inside a bbox and this bbox
// contains first intersecting volume.
TEST_F(BasicBihTest, inside_first)
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
        ref.intersect_surface = LocalSurfaceId{0};
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {0, 0, 0};
        auto result = this->get_result({pos, dir}, dist_map, infr);
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
        ref.intersect_surface = LocalSurfaceId{1};
        ref.hit_count = {2, 2, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
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
        ref.intersect_surface = LocalSurfaceId{2};
        ref.hit_count = {2, 2, 1};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
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
        ref.intersect_surface = LocalSurfaceId{3};
        ref.hit_count = {1, 1, 3};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V4 from the left
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.2}, {LocalVolumeId{5}, 1.3}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.intersect_surface = LocalSurfaceId{4};
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {2, 2, 2};
        auto result = this->get_result({pos, dir}, dist_map, infr);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray intersects V5 from the left
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{4}, 1.3}, {LocalVolumeId{5}, 1.2}};
    {
        IntersectResult ref;
        ref.distance = 1.2;
        ref.intersect_surface = LocalSurfaceId{5};
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, infr);
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
        ref.intersect_surface = LocalSurfaceId{5};
        ref.hit_count = {2, 2, 2};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 1.6);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Intersecting volume 5 from below
    pos = {4., -2., 50.};
    dir = {0., 1., 0.};
    dist_map = {
        {LocalVolumeId{3}, 2},
        {LocalVolumeId{4}, 1.5},
        {LocalVolumeId{5}, 1},
    };
    {
        IntersectResult ref;
        ref.distance = 1;
        ref.intersect_surface = LocalSurfaceId{5};
        ref.hit_count = {3, 3, 3};
        ref.miss_count = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 3);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

// Test the case where the first intersection does not yields the first volume
// collision
TEST_F(BasicBihTest, not_first)
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
        ref.intersect_surface = LocalSurfaceId{2};
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
        ref.intersect_surface = LocalSurfaceId{0};
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
        ref.intersect_surface = LocalSurfaceId{2};
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {3, 4, 4};
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
        ref.intersect_surface = LocalSurfaceId{2};
        ref.hit_count = {1, 1, 1};
        ref.miss_count = {3, 4, 4};
        auto result = this->get_result({pos, dir}, dist_map, 2.1);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

/*!
 * Test a large number of adjacent bounding boxes along the Z axis.
 *
 * The extents of the problem are {-1, -1, 0}, {1, 1, num_boxes}. Bounding
 * boxes less with even indices (`i % 2 == 0`) have pretend volumes in the
 * range z=[0.5, 1] offset by the box.
 */
class KebabTest : public BIHIntersectingVolFinderTest
{
  public:
    VecSetup make_bih_setups() const override
    {
        VecSetup result;
        if (false)
        {
            // FIXME: depth limit is forced cutoff: tree is unbalanced
            for (auto depth_limit : range(8))
            {
                inp::BIHBuilder setup;
                setup.depth_limit = 1 + 4 * depth_limit;
                result.push_back(setup);
            }
        }
        else
        {
            for (auto leaf_size : {1, 2, 4, 8, 12, 16, 20, 24})
            {
                inp::BIHBuilder setup;
                setup.max_leaf_size = leaf_size;
                result.push_back(setup);
            }
            return result;
        }
        return result;
    }

    static constexpr size_type num_boxes{1024};

    VecBBox make_bboxes() const override
    {
        using FastReal3 = FastBBox::Real3;
        VecBBox result;
        result.reserve(num_boxes);
        for (auto i : range(num_boxes))
        {
            result.emplace_back(FastReal3{-1, -1, i}, FastReal3{1, 1, i + 1});
        }
        return result;
    }
};

TEST_F(KebabTest, DISABLED_tree_output)
{
    for (auto&& s : this->get_bih_json_strings())
    {
        cout << "R\"json(" << s << ")json\"\n\n\n";
    }
}

// Test the case where the ray starts somewhere inside a bbox and this bbox
// contains first intersecting volume.
TEST_F(KebabTest, all)
{
    Real3 pos{0, 0, 0}, dir{0, 0, 1};
    DistMap dist_map;
    {
        SCOPED_TRACE("Test everything, no hits");
        auto result = this->get_result({pos, dir}, dist_map, infr);

        IntersectResult ref;
        ref.distance = inf;
        ref.intersect_surface = {};
        ref.hit_count = {0, 0, 0, 0, 0, 0, 0, 0};
        ref.miss_count = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
        EXPECT_REF_EQ(ref, result) << result;
    }
    {
        SCOPED_TRACE("Start halfway");
        pos = {0, 0, 512};
        auto result = this->get_result({pos, dir}, dist_map, infr);

        IntersectResult ref;
        ref.distance = inf;
        ref.intersect_surface = {};
        ref.hit_count = {0, 0, 0, 0, 0, 0, 0, 0};
        ref.miss_count = {513, 513, 513, 513, 513, 513, 513, 513};
        EXPECT_REF_EQ(ref, result) << result;
    }
    {
        SCOPED_TRACE("Start halfway, hit quickly");
        pos = {0, 0, 512};
        dist_map = {
            {LocalVolumeId{514}, 2.9},
            {LocalVolumeId{1000}, 488.9},
        };
        auto result = this->get_result({pos, dir}, dist_map, infr);

        IntersectResult ref;
        ref.distance = 2.9;
        ref.intersect_surface = LocalSurfaceId{514};
        ref.hit_count = {1, 1, 1, 1, 1, 1, 1, 1};
        ref.miss_count = {3, 3, 3, 3, 3, 3, 3, 3};
        EXPECT_REF_EQ(ref, result) << result;
    }
    {
        // Here we test closer leaves first, so we get closer hits
        // first and can eliminate further nodes
        SCOPED_TRACE("Start halfway, hit less quickly");
        pos = {0, 0, 512};
        dir = {0, 0, -1};
        dist_map = {
            {LocalVolumeId{510}, 2.1},
            {LocalVolumeId{500}, 12.1},
            {LocalVolumeId{400}, 112.1},
            {LocalVolumeId{200}, 312.1},
            {LocalVolumeId{0}, 512.1},
        };
        auto result = this->get_result({pos, dir}, dist_map, infr);

        IntersectResult ref;
        ref.distance = 2.1;
        ref.intersect_surface = LocalSurfaceId{510ul};
        ref.hit_count = {1, 1, 1, 1, 1, 2, 2, 2};
        ref.miss_count = {3, 3, 4, 8, 8, 15, 15, 15};
        EXPECT_REF_EQ(ref, result) << result;
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
