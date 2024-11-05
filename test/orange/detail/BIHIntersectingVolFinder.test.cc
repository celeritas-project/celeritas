//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/BIHIntersectingVolFinder.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "orange/detail/BIHBuilder.hh"
#include "orange/detail/BIHData.hh"
#include "orange/univ/detail/Types.hh"
#include "celeritas/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
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
class BIHIntersectingVolFinderTest : public Test
{
  public:
    using BIHBuilder = celeritas::detail::BIHBuilder;
    using BIHIntersectingVolFinder
        = celeritas::detail::BIHIntersectingVolFinder;
    using Intersection = celeritas::detail::Intersection;
    using Ray = celeritas::detail::BIHIntersectingVolFinder::Ray;
    using DistMap = std::map<LocalVolumeId, real_type>;

    struct MockVisitVol
    {
        detail::Intersection operator()(LocalVolumeId const& vol_id)
        {
            detail::OnLocalSurface on_surface{
                LocalSurfaceId{vol_id.unchecked_get()}, Sense::outside};
            return detail::Intersection{on_surface, distances[vol_id]};
        }

        DistMap distances;
    };

    void SetUp()
    {
        bboxes_.push_back(FastBBox::from_infinite());
        bboxes_.push_back({{0, 0, 0}, {1.6f, 1, 100}});
        bboxes_.push_back({{1.2f, 0, 0}, {2.8f, 1, 100}});
        bboxes_.push_back({{2.8f, 0, 0}, {5, 1, 100}});
        bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
        bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

        BIHBuilder builder(&storage_);
        bih_tree_ = builder(std::move(bboxes_));

        ref_storage_ = storage_;
    }

    void check_result(Ray const& ray,
                      DistMap const& distances,
                      LocalVolumeId vol_id,
                      real_type dist)
    {
        MockVisitVol visit_vol;
        visit_vol.distances = distances;

        auto find_volume = BIHIntersectingVolFinder(bih_tree_, ref_storage_);
        auto intersection = find_volume(ray, visit_vol);

        EXPECT_SOFT_EQ(dist, intersection.distance);
        EXPECT_EQ(vol_id.unchecked_get(),
                  intersection.surface.id().unchecked_get());
    }

  protected:
    static constexpr auto inff_
        = std::numeric_limits<fast_real_type>::infinity();
    std::vector<FastBBox> bboxes_;
    detail::BIHTree bih_tree_;
    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;
};

TEST_F(BIHIntersectingVolFinderTest, basic)
{
    {
        // Ray goes straight to V3
        Ray ray{{6, 0.5, 50.}, {-1., 0., 0.}};
        DistMap distances{{LocalVolumeId{0}, inff_},
                          {LocalVolumeId{1}, 4.4},
                          {LocalVolumeId{2}, 3.2},
                          {LocalVolumeId{3}, 1.},
                          {LocalVolumeId{4}, inff_},
                          {LocalVolumeId{5}, inff_}};

        this->check_result(ray, distances, LocalVolumeId{3}, 1.0);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
