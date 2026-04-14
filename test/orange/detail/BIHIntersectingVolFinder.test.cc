//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHIntersectingVolFinder.hh"

#include "orange/detail/BIHBuilder.hh"
#include "orange/univ/detail/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

// Mock class with operator() to serve as a visit_vol functor
class MockIntersector
{
  public:
    using Intersection = detail::Intersection;
    using DistMap = std::map<LocalVolumeId, real_type>;

  public:
    explicit MockIntersector(DistMap const& dist_map) : dist_map_(dist_map) {}

    Intersection operator()(LocalVolumeId const& vol_id, real_type max_distance)
    {
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

        detail::OnLocalSurface on_surface{
            LocalSurfaceId{vol_id.unchecked_get()}, Sense::outside};
        ++hits_;
        return Intersection{on_surface, iter->second};
    }

    size_type hits() const { return hits_; }
    size_type misses() const { return misses_; }

  private:
    DistMap const& dist_map_;
    size_type hits_{0};
    size_type misses_{0};
};

struct IntersectResult
{
    real_type distance{};
    LocalVolumeId vol_id{};
    std::vector<int> hits;
    std::vector<int> misses;
};

std::ostream& operator<<(std::ostream& os, IntersectResult const& ref)
{
    // clang-format off
    os << "/*** INTERSECT RESULT ***/\n"
          "IntersectResult ref;\n"
       << CELER_REF_ATTR(distance)
       << CELER_REF_ATTR(vol_id)
       << CELER_REF_ATTR(hits)
       << CELER_REF_ATTR(misses)
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
    IRE_COMPARE(vol_id);
    IRE_COMPARE(hits);
    IRE_COMPARE(misses);

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
class BihTreeIntersector
{
  public:
    using VecBBox = detail::BIHBuilder::VecBBox;
    using Ray = detail::BIHIntersectingVolFinder::Ray;

    BihTreeIntersector(VecBBox bboxes, detail::BIHBuilder::Input input)
    {
        detail::BIHBuilder build(&storage_, input);
        detail::BIHBuilder::SetLocalVolId implicit_vol_ids;
        bih_tree_ = build(std::move(bboxes), implicit_vol_ids);
        ref_storage_ = storage_;
    }

    template<class F>
    detail::Intersection operator()(Ray ray, F&& visit_vol) const
    {
        detail::BIHIntersectingVolFinder find_volume{bih_tree_, ref_storage_};
        return find_volume(ray, std::forward<F>(visit_vol));
    }

    template<class F>
    detail::Intersection
    operator()(Ray ray, F&& visit_vol, real_type max_dist) const
    {
        detail::BIHIntersectingVolFinder find_volume{bih_tree_, ref_storage_};
        return find_volume(ray, std::forward<F>(visit_vol), max_dist);
    }

  private:
    detail::BIHTreeRecord bih_tree_;
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
class BIHIntersectingVolFinderTest : public Test
{
  public:
    using Ray = BihTreeIntersector::Ray;
    using DistMap = MockIntersector::DistMap;

  protected:
    void SetUp() override
    {
        BihTreeIntersector::VecBBox bboxes = {
            FastBBox::from_infinite(),
            {{0, 0, 0}, {1.6f, 1, 100}},
            {{1.2f, 0, 0}, {2.8f, 1, 100}},
            {{2.8f, 0, 0}, {5, 1, 100}},
            {{0, -1, 0}, {5, 0, 100}},
            {{0, -1, 0}, {5, 0, 100}},
        };

        intersectors_.reserve(3);
        for (auto leaf_size : range(size_type{1}, size_type{4}))
        {
            intersectors_.emplace_back(bboxes,
                                       detail::BIHBuilder::Input{leaf_size});
        }
    }

    // Get results for a ray across all leaf-size intersectors
    IntersectResult get_result(Ray ray, DistMap const& dist_map)
    {
        IntersectResult result;
        for (auto& intersector : intersectors_)
        {
            MockIntersector visit_vol{dist_map};
            auto intersection = intersector(ray, visit_vol);
            if (result.hits.empty())
            {
                result.distance = intersection.distance;
                if (intersection)
                {
                    result.vol_id = LocalVolumeId{
                        intersection.surface.id().unchecked_get()};
                }
            }
            result.hits.push_back(static_cast<int>(visit_vol.hits()));
            result.misses.push_back(static_cast<int>(visit_vol.misses()));
        }
        return result;
    }

    // Get results for a ray across all leaf-size intersectors, with a max
    // search distance
    IntersectResult
    get_result(Ray ray, DistMap const& dist_map, real_type max_search_dist)
    {
        IntersectResult result;
        for (auto& intersector : intersectors_)
        {
            MockIntersector visit_vol{dist_map};
            auto intersection = intersector(ray, visit_vol, max_search_dist);
            if (result.hits.empty())
            {
                result.distance = intersection.distance;
                if (intersection)
                {
                    result.vol_id = LocalVolumeId{
                        intersection.surface.id().unchecked_get()};
                }
            }
            result.hits.push_back(static_cast<int>(visit_vol.hits()));
            result.misses.push_back(static_cast<int>(visit_vol.misses()));
        }
        return result;
    }

    std::vector<BihTreeIntersector> intersectors_;
};

// Test the case where the ray starts outside the bbox and the first bbox
// intersection yields the first volume intersection.
TEST_F(BIHIntersectingVolFinderTest, outside_first)
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
        ref.vol_id = LocalVolumeId{1};
        ref.hits = {1, 1, 1};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{2};
        ref.hits = {1, 1, 1};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{3};
        ref.hits = {3, 3, 3};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{4};
        ref.hits = {1, 1, 1};
        ref.misses = {2, 2, 2};
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
        ref.vol_id = LocalVolumeId{5};
        ref.hits = {2, 2, 2};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{};
        ref.hits = {0, 0, 0};
        ref.misses = {3, 3, 3};
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
        ref.vol_id = LocalVolumeId{5};
        ref.hits = {2, 2, 2};
        ref.misses = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 1.3);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

// Test the case where the ray starts somewhere inside a bbox and this bbox
// contains first intersecting volume.
TEST_F(BIHIntersectingVolFinderTest, inside_first)
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
        ref.vol_id = LocalVolumeId{0};
        ref.hits = {2, 2, 2};
        ref.misses = {0, 0, 0};
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
        ref.vol_id = LocalVolumeId{1};
        ref.hits = {1, 1, 1};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{2};
        ref.hits = {1, 1, 1};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{3};
        ref.hits = {3, 3, 3};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{4};
        ref.hits = {1, 1, 1};
        ref.misses = {2, 2, 2};
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
        ref.vol_id = LocalVolumeId{5};
        ref.hits = {2, 2, 2};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{};
        ref.hits = {0, 0, 0};
        ref.misses = {3, 3, 3};
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
        ref.vol_id = LocalVolumeId{5};
        ref.hits = {2, 2, 2};
        ref.misses = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map, 1.6);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

// Test the case where the first intersection does not yields the first volume
// collision
TEST_F(BIHIntersectingVolFinderTest, not_first)
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
        ref.vol_id = LocalVolumeId{2};
        ref.hits = {2, 2, 2};
        ref.misses = {1, 1, 1};
        auto result = this->get_result({pos, dir}, dist_map);
        EXPECT_REF_EQ(ref, result) << result;
    }

    // Ray goes all the way through V1, V2 and V3, intersects V0
    pos = {-0.5, 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, 11.}

    };
    {
        IntersectResult ref;
        ref.distance = 11.;
        ref.vol_id = LocalVolumeId{0};
        ref.hits = {1, 1, 1};
        ref.misses = {3, 3, 3};
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
        ref.vol_id = LocalVolumeId{2};
        ref.hits = {1, 1, 1};
        ref.misses = {4, 4, 4};
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
        ref.vol_id = LocalVolumeId{};
        ref.hits = {0, 0, 0};
        ref.misses = {1, 1, 1};
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
        ref.vol_id = LocalVolumeId{2};
        ref.hits = {1, 1, 1};
        ref.misses = {4, 4, 4};
        auto result = this->get_result({pos, dir}, dist_map, 2.1);
        EXPECT_REF_EQ(ref, result) << result;
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
