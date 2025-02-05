//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.test.cc
//---------------------------------------------------------------------------//

#include "orange/detail/BIHIntersectingVolFinder.hh"

#include "orange/detail/BIHBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/* The BIHIntersectingVolFinder class is tested with the following geometry,
 * consisting of partial and fully overlapping bounding boxes.
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
class BIHIntersectingVolFinderTest : public Test
{
  public:
    using BIHBuilder = celeritas::detail::BIHBuilder;
    using BIHIntersectingVolFinder
        = celeritas::detail::BIHIntersectingVolFinder;
    using Intersection = celeritas::detail::Intersection;
    using Ray = celeritas::detail::BIHIntersectingVolFinder::Ray;
    using DistMap = std::map<LocalVolumeId, real_type>;

    void SetUp()
    {
        bboxes_.push_back(FastBBox::from_infinite());
        bboxes_.push_back({{0, 0, 0}, {1.6f, 1, 100}});
        bboxes_.push_back({{1.2f, 0, 0}, {2.8f, 1, 100}});
        bboxes_.push_back({{2.8f, 0, 0}, {5, 1, 100}});
        bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
        bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

        BIHBuilder build(&storage_);
        BIHBuilder::SetLocalVolId implicit_vol_ids_;
        bih_tree_ = build(std::move(bboxes_), implicit_vol_ids_);
        ref_storage_ = storage_;
    }

  protected:
    // Mock class with operator() to serve as a visit_vol functor
    struct MockVisitVol
    {
        detail::Intersection operator()(LocalVolumeId const& vol_id,
                                        real_type max_search_dist
                                        [[maybe_unused]])
        {
            detail::OnLocalSurface on_surface{
                LocalSurfaceId{vol_id.unchecked_get()}, Sense::outside};
            return detail::Intersection{on_surface, dist_map[vol_id]};
        }

        DistMap dist_map;
    };

    // Check result for a single ray
    void check_result(Ray ray,
                      DistMap const& dist_map,
                      LocalVolumeId vol_id,
                      real_type dist)
    {
        MockVisitVol visit_vol;
        visit_vol.dist_map = dist_map;

        auto find_volume = BIHIntersectingVolFinder(bih_tree_, ref_storage_);
        auto intersection = find_volume(ray, visit_vol);

        EXPECT_SOFT_EQ(dist, intersection.distance);
        EXPECT_EQ(vol_id.unchecked_get(),
                  intersection.surface.id().unchecked_get());
    }

    // Check result for a single ray, with a max search distance
    void check_result(Ray ray,
                      DistMap const& dist_map,
                      LocalVolumeId vol_id,
                      real_type max_search_dist,
                      real_type dist)
    {
        MockVisitVol visit_vol;
        visit_vol.dist_map = dist_map;

        auto find_volume = BIHIntersectingVolFinder(bih_tree_, ref_storage_);
        auto intersection = find_volume(ray, visit_vol, max_search_dist);

        EXPECT_SOFT_EQ(dist, intersection.distance);
        EXPECT_EQ(vol_id.unchecked_get(),
                  intersection.surface.id().unchecked_get());
    }

    static constexpr auto inff_
        = std::numeric_limits<fast_real_type>::infinity();
    std::vector<FastBBox> bboxes_;
    detail::BIHTree bih_tree_;
    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;
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
    dist_map = {{LocalVolumeId{0}, 10},
                {LocalVolumeId{1}, 1},
                {LocalVolumeId{2}, 1.2},
                {LocalVolumeId{3}, 2.8},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{1}, 1.0);

    // Ray intersects V2 from above
    pos = {2., 2., 50.};
    dir = {0., -1., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, 1.},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 2.},
                {LocalVolumeId{5}, 2.}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{2}, 1);

    // Ray intersects V3 from the right
    pos = {6, 0.5, 50.};
    dir = {-1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, 4.4},
                {LocalVolumeId{2}, 3.2},
                {LocalVolumeId{3}, 1.},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{3}, 1.0);

    // Ray intersects V4 from the left
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.2},
                {LocalVolumeId{5}, 1.3}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{4}, 1.2);

    // Ray intersects V5 from the left
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.3},
                {LocalVolumeId{5}, 1.2}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{5}, 1.2);

    // Ray intersects V5 from the left, max search distance is closer
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.3},
                {LocalVolumeId{5}, 1.2}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{}, 1.1, 1.1);

    // Ray intersects V5 from the left, max search distance is further
    pos = {-0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.3},
                {LocalVolumeId{5}, 1.2}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{5}, 1.3, 1.2);
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
    dist_map = {{LocalVolumeId{0}, 0.5},
                {LocalVolumeId{1}, 1},
                {LocalVolumeId{2}, 1.2},
                {LocalVolumeId{3}, 2.8},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{0}, 0.5);

    // Ray starts in V1 and intersects V1
    pos = {1., 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, 10},
                {LocalVolumeId{1}, 0.1},
                {LocalVolumeId{2}, 0.7},
                {LocalVolumeId{3}, 2.3},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{1}, 0.1);

    // Ray starts in V2 and intersects V2
    pos = {2., 2., 50.};
    dir = {0., -1., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, 1.},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 2.},
                {LocalVolumeId{5}, 2.}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{2}, 1);

    // Ray starts in V3 and intersects V3
    pos = {4, 0.5, 50.};
    dir = {-1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, 2.4},
                {LocalVolumeId{2}, 1.2},
                {LocalVolumeId{3}, 1.},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{3}, 1.0);

    // Ray intersects V4 from the left
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.2},
                {LocalVolumeId{5}, 1.3}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{4}, 1.2);

    // Ray intersects V5 from the left
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.3},
                {LocalVolumeId{5}, 1.2}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{5}, 1.2);

    // Ray intersects V5 from the left, max search distance is closer
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.3},
                {LocalVolumeId{5}, 1.2}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{}, 0.1, 0.1);

    // Ray intersects V5 from the left, max search distance is further
    pos = {0.5, -0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, 1.3},
                {LocalVolumeId{5}, 1.2}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{5}, 1.6, 1.2);
}

// Test the case where the first intersection does not yeilds the first volume
// collision
TEST_F(BIHIntersectingVolFinderTest, not_first)
{
    Real3 pos, dir;
    DistMap dist_map;

    // Ray goes through V1 but intersects with V2 first
    pos = {-0.5, 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, 2.0},
                {LocalVolumeId{2}, 1.7},
                {LocalVolumeId{3}, 3.3},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{2}, 1.7);

    // Ray goes all the way through V1, V2 and V3, interects V0
    pos = {-0.5, 0.5, 50.};
    dir = {1., 0., 0.};
    dist_map = {{LocalVolumeId{0}, 11.},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, inff_},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{0}, 11.);

    // Ray goes through V4 and V5 and intersects with V2
    pos = {1.5, -2, 50.};
    dir = {0., 1., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, 1.5},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{2}, 1.5);

    // Ray goes through V4 and V5 and intersects with V2, max search is closer
    pos = {1.5, -2, 50.};
    dir = {0., 1., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, 1.5},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{}, 0.8, 0.8);

    // Ray goes through V4 and V5 and intersects with V2, max search is further
    pos = {1.5, -2, 50.};
    dir = {0., 1., 0.};
    dist_map = {{LocalVolumeId{0}, inff_},
                {LocalVolumeId{1}, inff_},
                {LocalVolumeId{2}, 1.5},
                {LocalVolumeId{3}, inff_},
                {LocalVolumeId{4}, inff_},
                {LocalVolumeId{5}, inff_}};
    this->check_result({pos, dir}, dist_map, LocalVolumeId{2}, 2.1, 1.5);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
