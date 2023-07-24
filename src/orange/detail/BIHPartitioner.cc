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
 * Determine if a set of bounding boxes can be partitioned
 */
bool BIHPartitioner::is_partitionable(VecIndices const& indices) const
{
    auto centers = this->axes_centers(indices);
    return std::any_of(centers.begin(),
                       centers.end(),
                       [](std::vector<real_type> const& centers) {
                           return centers.size() > 1;
                       });
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
        auto ax = to_int(axis);

        if (axes_centers[ax].size() > 1)
        {
            partition.axis = axis;
            auto size = axes_centers[ax].size();
            partition.location
                = (axes_centers[ax][size / 2 - 1] + axes_centers[ax][size / 2])
                  / 2;
            break;
        }
    }

    if (partition.axis != Axis::size_)
    {
        this->apply_partition(indices, partition);
    }

    CELER_EXPECT(partition);
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
/*!
 * Divide bboxes into left and right branches based on a partition.
 */
void BIHPartitioner::apply_partition(VecIndices const& indices,
                                     Partition& p) const
{
    CELER_EXPECT(!indices.empty());

    for (auto i : range(indices.size()))
    {
        if (centers_->at(indices[i].unchecked_get())[to_int(p.axis)]
            < p.location)
        {
            p.left_indices.push_back(indices[i]);
        }
        else
        {
            p.right_indices.push_back(indices[i]);
        }
    }

    p.left_bbox = bbox_union(*bboxes_, p.left_indices);
    p.right_bbox = bbox_union(*bboxes_, p.right_indices);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
