//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHPartitioner.cc
//---------------------------------------------------------------------------//
#include "BIHPartitioner.hh"

#include "BoundingBoxUtils.hh"

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Sort and uniquify a vector.
 */
void sort_and_uniquify(std::vector<celeritas::real_type>& vec)
{
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
}
}  // namespace

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and respective centers.
 */
BIHPartitioner::BIHPartitioner(VecBBox* bboxes, VecReal3* centers)
    : bboxes_(bboxes), centers_(centers)
{
    CELER_EXPECT(!bboxes_->empty());
    CELER_EXPECT(bboxes_->size() == centers_->size());
}

//---------------------------------------------------------------------------//
/*!
 * Find a suitable partition for the given bounding boxes.
 *
 * If no suitable partition is found an empty Partition object is returned.
 */
BIHPartitioner::Partition
BIHPartitioner::operator()(VecIndices const& indices) const
{
    auto sorted_axes = sort_axes(bbox_union(*bboxes_, indices));
    auto axes_centers = this->axes_centers(indices);

    Partition partition;

    for (Axis axis : sorted_axes)
    {
        printf("CHECKING Parition on AXIS %i\n", to_int(axis));
        auto ax = to_int(axis);

        if (axes_centers[ax].size() > 1)
        {
            partition.axis = axis;
            auto size = axes_centers[ax].size();
            partition.location
                = (axes_centers[ax][size / 2 - 1] + axes_centers[ax][size / 2])
                  / 2;
            printf("\tChoosing partition location %f\n", partition.location);
            break;
        }
    }
    return partition;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Create sorted and uniquified X, Y, Z values of bbox centers.
 */
BIHPartitioner::AxesCenters
BIHPartitioner::axes_centers(VecIndices const& indices) const
{
    AxesCenters axes_centers{{}, {}, {}};

    for (auto id : indices)
    {
        Real3 center = centers_->at(id.unchecked_get());
        for (auto axis : range(Axis::size_))
        {
            auto ax = to_int(axis);
            axes_centers[ax].push_back(center[ax]);
        }
    }

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);
        sort_and_uniquify(axes_centers[ax]);
    }

    return axes_centers;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
