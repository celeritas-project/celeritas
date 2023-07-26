//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHPartitioner.cc
//---------------------------------------------------------------------------//
#include "BIHPartitioner.hh"

#include "orange/BoundingBoxUtils.hh"

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
 * Find a suitable partition for the given bounding boxes.
 *
 * If no partition is found, an empty partition is return
 */
BIHPartitioner::Partition
BIHPartitioner::operator()(VecIndices const& indices) const
{
    CELER_EXPECT(!indices.empty());

    Partition best_partition;
    real_type best_cost = std::numeric_limits<real_type>::infinity();

    auto axes_centers = this->calc_axes_centers(indices);

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);

        for (auto i : range(axes_centers[ax].size() - 1))
        {
            Partition p;
            p.axis = axis;
            p.position = (axes_centers[ax][i] + axes_centers[ax][i + 1]) / 2;
            this->apply_partition(indices, p);
            auto cost = this->calc_cost(p);

            if (cost < best_cost)
            {
                best_partition = std::move(p);
                best_cost = cost;
            }
        }
    }

    return best_partition;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Create sorted and uniquified X, Y, Z values of bbox centers.
 */
BIHPartitioner::AxesCenters
BIHPartitioner::calc_axes_centers(VecIndices const& indices) const
{
    CELER_EXPECT(!indices.empty());

    AxesCenters axes_centers{{}, {}, {}};

    for (auto id : indices)
    {
        CELER_ASSERT(id < centers_->size());
        Real3 center = (*centers_)[id.unchecked_get()];
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
        CELER_ASSERT(indices[i] < centers_->size());
        if ((*centers_)[indices[i].unchecked_get()][to_int(p.axis)]
            < p.position)
        {
            p.indices[BIHInnerNode::Edge::left].push_back(indices[i]);
        }
        else
        {
            p.indices[BIHInnerNode::Edge::right].push_back(indices[i]);
        }
    }

    p.bboxes[BIHInnerNode::Edge::left]
        = bbox_union(*bboxes_, p.indices[BIHInnerNode::Edge::left]);
    p.bboxes[BIHInnerNode::Edge::right]
        = bbox_union(*bboxes_, p.indices[BIHInnerNode::Edge::right]);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cost of partition using a surface area heuristic.
 */
real_type BIHPartitioner::calc_cost(Partition const& p) const
{
    CELER_EXPECT(p);

    return surface_area(p.bboxes[BIHInnerNode::Edge::left])
               * p.indices[BIHInnerNode::Edge::left].size()
           + surface_area(p.bboxes[BIHInnerNode::Edge::right])
                 * p.indices[BIHInnerNode::Edge::right].size();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
