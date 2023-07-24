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
        real_type location = std::numeric_limits<real_type>::infinity();

        VecIndices left_indices;
        VecIndices right_indices;

        BoundingBox left_bbox;
        BoundingBox right_bbox;

        explicit operator bool() const
        {
            return axis != Axis::size_ && std::isfinite(location)
                   && !left_indices.empty() && !right_indices.empty()
                   && left_bbox && right_bbox;
        }
    };

    //!@}

  public:
    //! Default constructor
    BIHPartitioner() {}

    // Construct from vector of bounding boxes and respective centers.
    explicit BIHPartitioner(VecBBox* bboxes, VecReal3* centers);

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

    //// HELPER FUNCTIONS ////

    // Create sorted and uniquified X, Y, Z values of bbox centers
    AxesCenters axes_centers(VecIndices const& indices) const;

    // Divide bboxes into left and right branches based on a partition.
    void apply_partition(VecIndices const& indices, Partition& partition) const;

    real_type calc_cost(Partition const& partition) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
