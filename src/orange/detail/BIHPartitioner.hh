//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHPartitioner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "orange/BoundingBox.hh"
#include "orange/OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Partition bounding boxes using a surface area heuristic.
 *
 * The class take a vector of bounding boxes as an input, and outputs a
 * Partition object describing the optional partition. To find the optimal
 * partition, all possible candidate partitions along the x, y, and z axis are
 * evaluated using a cost function. The cost function is based on a standard
 * surface area heuristic.
 */
class BIHPartitioner
{
  public:
    //!@{
    //! \name Type aliases
    using VecBBox = std::vector<BoundingBox>;
    using VecReal3 = std::vector<Real3>;
    using VecIndices = std::vector<LocalVolumeId>;

    struct Partition
    {
        Axis axis = Axis::size_;
        real_type position = std::numeric_limits<real_type>::infinity();

        VecIndices left_indices;
        VecIndices right_indices;

        BoundingBox left_bbox;
        BoundingBox right_bbox;

        explicit operator bool() const
        {
            return axis != Axis::size_ && std::isfinite(position)
                   && !left_indices.empty() && !right_indices.empty()
                   && left_bbox && right_bbox;
        }
    };

    //!@}

  public:
    //! Default constructor
    BIHPartitioner() = default;

    // Construct from vector of bounding boxes and respective centers.
    explicit BIHPartitioner(VecBBox* bboxes, VecReal3* centers);

    explicit inline operator bool() { return bboxes_ != nullptr; }

    // Determine is a set of bounding boxes can be partitioned
    bool is_partitionable(VecIndices const& indices) const;

    // Find a suitable partition for the given bounding boxes
    Partition operator()(VecIndices const& indicies) const;

  private:
    /// TYPES ///
    using AxesCenters = std::vector<std::vector<real_type>>;

    //// DATA ////
    VecBBox* bboxes_;
    VecReal3* centers_;
    static constexpr real_type uniquify_tol_ = 1E-12;

    //// HELPER FUNCTIONS ////

    // Create sorted and uniquified X, Y, Z values of bbox centers
    AxesCenters axes_centers(VecIndices const& indices) const;

    // Divide bboxes into left and right branches based on a partition
    void apply_partition(VecIndices const& indices, Partition& partition) const;

    // Calculate the cost of partition using a surface area heuristic
    real_type calc_cost(Partition const& partition) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
