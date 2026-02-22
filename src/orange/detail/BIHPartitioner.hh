//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHPartitioner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/EnumArray.hh"
#include "geocel/BoundingBox.hh"

#include "../OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Partition bounding boxes using a surface area heuristic.
 *
 * This class takes a vector of bounding boxes as an input, and outputs a
 * Partition object describing the optional partition (among those tested). To
 * find the optimal partition, candidate partitions along the x, y, nd z axis
 * are evaluated using a cost function. The cost function is based on a
 * standard surface area heuristic.
 */
class BIHPartitioner
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<fast_real_type, 3>;
    using VecBBox = std::vector<FastBBox>;
    using VecReal3 = std::vector<Real3>;
    using VecIndices = std::vector<LocalVolumeId>;
    using Side = BIHInnerNode::Side;

    //! Output struct specifying the indices and bboxes of the left and right
    //! sides of the partition
    struct Partition
    {
        Axis axis = Axis::size_;
        real_type position = std::numeric_limits<real_type>::infinity();

        EnumArray<Side, VecIndices> indices;
        EnumArray<Side, FastBBox> bboxes;

        explicit operator bool() const
        {
            return axis != Axis::size_ && std::isfinite(position)
                   && !indices[Side::left].empty()
                   && !indices[Side::right].empty() && bboxes[Side::left]
                   && bboxes[Side::right];
        }
    };
    //!@}

  public:
    //! Default constructor
    BIHPartitioner() = default;

    // Construct from all bounding bounding boxes in a universe
    explicit BIHPartitioner(VecBBox const* bboxes,
                            VecReal3 const* centers,
                            size_type num_part_cands);

    // Find a suitable partition for the given subset of bounding boxes
    Partition operator()(VecIndices const& indices) const;

    // True when assigned
    explicit inline operator bool() const { return bboxes_ != nullptr; }

  private:
    /// TYPES ///
    using AxesCenters = std::vector<std::vector<real_type>>;

    //// DATA ////

    //! All bounding boxes to be partitioned
    VecBBox const* bboxes_{nullptr};
    //! The centers of each bounding box
    VecReal3 const* centers_{nullptr};
    //! The number of partition candidates to check per axis
    size_type num_part_cands_{0};

    //// HELPER FUNCTIONS ////

    // Create sorted and uniquified x, y, z values of bbox centers
    AxesCenters calc_axes_centers(VecIndices const& indices) const;

    // Create a partition object
    Partition make_partition(VecIndices const& indices,
                             Axis axis,
                             real_type position) const;

    // Calculate the cost of partition using a surface area heuristic
    real_type calc_cost(Partition const& partition) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
