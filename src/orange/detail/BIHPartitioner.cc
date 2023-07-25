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
void sort_and_uniquify(std::vector<celeritas::real_type>& vec,
                       celeritas::real_type tol)
{
    std::sort(vec.begin(), vec.end());

    auto last = std::unique(vec.begin(),
                            vec.end(),
                            [&](celeritas::real_type a, celeritas::real_type b) {
                                return std::abs(a - b) < tol;
                            });

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
    CELER_EXPECT(!indices.empty());

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
 */
BIHPartitioner::Partition
BIHPartitioner::operator()(VecIndices const& indices) const
{
    CELER_EXPECT(!indices.empty());

    Partition best_partition;
    real_type best_cost = std::numeric_limits<real_type>::infinity();

    auto axes_centers = this->axes_centers(indices);

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);

        for (auto i : range(axes_centers[ax].size() - 1))
        {
            Partition p;
            p.axis = axis;
            p.location = (axes_centers[ax][i] + axes_centers[ax][i + 1]) / 2;
            this->apply_partition(indices, p);
            auto cost = this->calc_cost(p);

            if (cost < best_cost)
            {
                best_partition = p;
                best_cost = cost;
            }
        }
    }

    CELER_VALIDATE(best_partition, << "calculated partition not valid");

    return best_partition;
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
    CELER_EXPECT(!indices.empty());

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
        sort_and_uniquify(axes_centers[ax], uniquify_tol_);
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
/*!
 * Calculate the cost of partition using a surface area heuristic.
 */
real_type BIHPartitioner::calc_cost(Partition const& p) const
{
    CELER_EXPECT(p);

    return surface_area(p.left_bbox) * p.left_indices.size()
           + surface_area(p.right_bbox) * p.right_indices.size();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
