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
            return {};
            ++misses_;
        }

        if (iter->second > max_distance)
        {
            // Distance is outside the maximum
            return {};
            ++misses_;
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
    int hits{0};
    int misses{0};
    real_type distance{};
    LocalVolumeId vol_id{};
};

std::ostream& operator<<(std::ostream& os, IntersectResult const& ref)
{
    // clang-format off
    os << "/*** INTERSECT RESULT ***/\n"
          "IntersectResult ref;\n"
       << CELER_REF_ATTR(hits)
       << CELER_REF_ATTR(misses)
       << CELER_REF_ATTR(distance)
       << CELER_REF_ATTR(vol_id)
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

#undef IRE_COMPARE
    return result;
}

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
    using BIHBuilder = detail::BIHBuilder;
    using BIHIntersectingVolFinder = detail::BIHIntersectingVolFinder;
    using Ray = detail::BIHIntersectingVolFinder::Ray;
    using DistMap = MockIntersector::DistMap;

  protected:
    void setup(size_type max_leaf_size)
    {
        BIHBuilder::VecBBox bboxes = {
            FastBBox::from_infinite(),
            {{0, 0, 0}, {1.6f, 1, 100}},
            {{1.2f, 0, 0}, {2.8f, 1, 100}},
            {{2.8f, 0, 0}, {5, 1, 100}},
            {{0, -1, 0}, {5, 0, 100}},
            {{0, -1, 0}, {5, 0, 100}},
        };

        BIHBuilder build(&storage_, BIHBuilder::Input{max_leaf_size});
        BIHBuilder::SetLocalVolId implicit_vol_ids_;
        bih_tree_ = build(std::move(bboxes), implicit_vol_ids_);
        ref_storage_ = storage_;
    }

    // Get the result for a single ray
    IntersectResult get_result(Ray ray, DistMap const& dist_map)
    {
        MockIntersector visit_vol{dist_map};

        auto find_volume = BIHIntersectingVolFinder(bih_tree_, ref_storage_);
        auto intersection = find_volume(ray, visit_vol);

        IntersectResult result;
        result.hits = visit_vol.hits();
        result.misses = visit_vol.misses();
        result.distance = intersection.distance;
        if (intersection)
        {
            result.vol_id
                = LocalVolumeId{intersection.surface.id().unchecked_get()};
        }
        return result;
    }

    // Get the result for a single ray, with a max search distance
    IntersectResult
    get_result(Ray ray, DistMap const& dist_map, real_type max_search_dist)
    {
        MockIntersector visit_vol{dist_map};

        auto find_volume = BIHIntersectingVolFinder(bih_tree_, ref_storage_);
        auto intersection = find_volume(ray, visit_vol, max_search_dist);

        IntersectResult result;
        result.hits = visit_vol.hits();
        result.misses = visit_vol.misses();
        result.distance = intersection.distance;
        if (intersection)
        {
            result.vol_id
                = LocalVolumeId{intersection.surface.id().unchecked_get()};
        }
        return result;
    }

    std::vector<FastBBox> bboxes_;
    detail::BIHTreeRecord bih_tree_;
    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;
};

// Test the case where the ray starts outside the bbox and the first bbox
// intersection yields the first volume intersection.
TEST_F(BIHIntersectingVolFinderTest, outside_first)
{
    auto run_test = [&](size_type max_leaf_size) {
        this->setup(max_leaf_size);
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
            auto result = this->get_result({pos, dir}, dist_map, 1.3);
            EXPECT_REF_EQ(ref, result) << result;
        }
    };

    for (auto max_leaf_size : range(1, 4))
    {
        run_test(max_leaf_size);
    }
}

// Test the case where the ray starts somewhere inside a bbox and this bbox
// contains first intersecting volume.
TEST_F(BIHIntersectingVolFinderTest, inside_first)
{
    auto run_test = [&](size_type max_leaf_size) {
        this->setup(max_leaf_size);
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
            auto result = this->get_result({pos, dir}, dist_map);
            EXPECT_REF_EQ(ref, result) << result;
        }

        // Ray starts in V2 and intersects V2
        pos = {2., 2., 50.};
        dir = {0., -1., 0.};
        dist_map = {{LocalVolumeId{2}, 1.},
                    {LocalVolumeId{4}, 2.},
                    {LocalVolumeId{5}, 2.}};
        {
            IntersectResult ref;
            ref.distance = 1;
            ref.vol_id = LocalVolumeId{2};
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
            auto result = this->get_result({pos, dir}, dist_map, 1.6);
            EXPECT_REF_EQ(ref, result) << result;
        }
    };

    for (auto max_leaf_size : range(1, 4))
    {
        run_test(max_leaf_size);
    }
}

// Test the case where the first intersection does not yields the first volume
// collision
TEST_F(BIHIntersectingVolFinderTest, not_first)
{
    auto run_test = [&](size_type max_leaf_size) {
        this->setup(max_leaf_size);
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
            auto result = this->get_result({pos, dir}, dist_map, 2.1);
            EXPECT_REF_EQ(ref, result) << result;
        }
    };

    for (auto max_leaf_size : range(1, 4))
    {
        run_test(max_leaf_size);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
