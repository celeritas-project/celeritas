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

using BIHBuilder = celeritas::detail::BIHBuilder;
using BIHInnerNode = celeritas::detail::BIHInnerNode;
using BIHLeafNode = celeritas::detail::BIHLeafNode;
using BIHIntersectingVolFinder = celeritas::detail::BIHIntersectingVolFinder;
using Intersection = celeritas::detail::Intersection;
using Ray = celeritas::detail::BIHIntersectingVolFinder::Ray;

namespace celeritas
{
namespace test
{
class BIHIntersectingVolFinderTest : public Test
{
  public:
    // TYPES //
    struct MockVisitVol
    {
        using DistMap = std::map<LocalVolumeId, real_type>;

        MockVisitVol(DistMap distances) : distances_{distances} {};

        detail::Intersection operator()(LocalVolumeId const& vol_id)
        {
            detail::OnLocalSurface on_surface{
                LocalSurfaceId{vol_id.unchecked_get()}, Sense::outside};
            return detail::Intersection{on_surface, distances_[vol_id]};
        }

        DistMap distances_;
    };

    void SetUp() {}

  protected:
    static constexpr auto inff_
        = std::numeric_limits<fast_real_type>::infinity();
    std::vector<FastBBox> bboxes_;

    BIHTreeData<Ownership::value, MemSpace::host> storage_;
    BIHTreeData<Ownership::const_reference, MemSpace::host> ref_storage_;
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
TEST_F(BIHIntersectingVolFinderTest, basic)
{
    bboxes_.push_back(FastBBox::from_infinite());
    bboxes_.push_back({{0, 0, 0}, {1.6f, 1, 100}});
    bboxes_.push_back({{1.2f, 0, 0}, {2.8f, 1, 100}});
    bboxes_.push_back({{2.8f, 0, 0}, {5, 1, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});
    bboxes_.push_back({{0, -1, 0}, {5, 0, 100}});

    BIHBuilder bih(&storage_);
    auto bih_tree = bih(std::move(bboxes_));

    ref_storage_ = storage_;
    BIHIntersectingVolFinder find_volume(bih_tree, ref_storage_);

    {
        // Ray goes straight to V3
        Ray ray{{6, 0.5, 50.}, {-1., 0., 0.}};

        MockVisitVol::DistMap distances{{LocalVolumeId{0}, inff_},
                                        {LocalVolumeId{1}, 4.4},
                                        {LocalVolumeId{2}, 3.2},
                                        {LocalVolumeId{3}, 1.},
                                        {LocalVolumeId{4}, inff_},
                                        {LocalVolumeId{5}, inff_}};

        MockVisitVol visit_vol(distances);
        auto intersection = find_volume(ray, visit_vol);
        EXPECT_EQ(1.0, intersection.distance);
        EXPECT_EQ(3, intersection.surface.id().unchecked_get());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
