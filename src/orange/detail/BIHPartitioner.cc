//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHPartitioner.cc
//---------------------------------------------------------------------------//
#include "BIHPartitioner.hh"

#include "corecel/math/SoftEqual.hh"

#include "BIHUtils.hh"
#include "../BoundingBoxUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Sort and uniquify a vector.
 */
void sort_and_uniquify(std::vector<real_type>& vec)
{
    std::sort(vec.begin(), vec.end());

    celeritas::SoftEqual se;
    auto last = std::unique(vec.begin(), vec.end(), se);

    vec.erase(last, vec.end());
}
}  // namespace
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from all bounding bounding boxes in a universe.
 *
 * \param[in] bboxes          All bounding boxes the could be partitioned via
 *                            calls to operator()
 * \param[in] centers         The center position of each bounding box
 * \param[in] num_part_cands  The number of candidate partitions to check per
 *                            axis
 */
BIHPartitioner::BIHPartitioner(VecBBox const* bboxes,
                               VecReal3 const* centers,
                               size_type num_part_cands)
    : bboxes_(bboxes), centers_(centers), num_part_cands_(num_part_cands)
{
    CELER_EXPECT(!bboxes_->empty());
    CELER_EXPECT(bboxes_->size() == centers_->size());
    CELER_EXPECT(num_part_cands_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Find a suitable partition for the given subset of bounding boxes.
 *
 * If no partition is found, an empty partition is returned
 */
BIHPartitioner::Partition
BIHPartitioner::operator()(VecIndices const& indices) const
{
    CELER_EXPECT(*this);

    Partition best_partition;
    real_type best_cost = std::numeric_limits<real_type>::infinity();

    auto axes_centers = this->calc_axes_centers(indices);

    for (auto axis : range(Axis::size_))
    {
        auto ax = to_int(axis);

        // Loop through <candidates_per_axis_> equally-spaced partition
        // candidates

        auto step_size
            = std::max(static_cast<size_type>(axes_centers[ax].size()
                                              / (num_part_cands_ + 1)),
                       size_type{1});

        for (auto i = step_size; i < axes_centers[ax].size(); i += step_size)
        {
            auto position = (axes_centers[ax][i - 1] + axes_centers[ax][i]) / 2;

            auto p = this->make_partition(indices, axis, position);
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
        for (auto ax : range(to_int(Axis::size_)))
        {
            axes_centers[ax].push_back(center[ax]);
        }
    }

    for (auto ax : range(to_int(Axis::size_)))
    {
        sort_and_uniquify(axes_centers[ax]);
    }

    return axes_centers;
}

//---------------------------------------------------------------------------//
/*!
 * Divide bboxes into left and right branches based on a partition.
 */
BIHPartitioner::Partition
BIHPartitioner::make_partition(VecIndices const& indices,
                               Axis axis,
                               real_type position) const
{
    CELER_EXPECT(indices.size() > 1);

    Partition p;
    p.axis = axis;
    p.position = position;

    for (auto i : range(indices.size()))
    {
        CELER_ASSERT(indices[i] < centers_->size());
        if ((*centers_)[indices[i].unchecked_get()][to_int(p.axis)]
            < p.position)
        {
            p.indices[Side::left].push_back(indices[i]);
        }
        else
        {
            p.indices[Side::right].push_back(indices[i]);
        }
    }

    CELER_ASSERT(!p.indices[Side::left].empty());
    CELER_ASSERT(!p.indices[Side::right].empty());

    p.bboxes[Side::left] = calc_union(*bboxes_, p.indices[Side::left]);
    p.bboxes[Side::right] = calc_union(*bboxes_, p.indices[Side::right]);

    CELER_ENSURE(p);
    return p;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the cost of partition using a surface area heuristic.
 */
real_type BIHPartitioner::calc_cost(Partition const& p) const
{
    CELER_EXPECT(p);

    return calc_surface_area(p.bboxes[Side::left])
               * p.indices[Side::left].size()
           + calc_surface_area(p.bboxes[Side::right])
                 * p.indices[Side::right].size();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
