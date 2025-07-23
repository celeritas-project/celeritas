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
    using Real3 = Array<fast_real_type, 3>;
    using VecBBox = std::vector<FastBBox>;
    using VecReal3 = std::vector<Real3>;
    using VecIndices = std::vector<LocalVolumeId>;
    using Side = BIHInnerNode::Side;

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

    // Construct from vector of bounding boxes and respective centers.
    explicit BIHPartitioner(VecBBox const* bboxes, VecReal3 const* centers);

    explicit inline operator bool() const
    {
        return bboxes_ != nullptr && centers_ != nullptr;
    }

    // Find a suitable partition for the given bounding boxes
    Partition operator()(VecIndices const& indices) const;

  private:
    /// TYPES ///
    using AxesCenters = std::vector<std::vector<real_type>>;

    //// DATA ////
    VecBBox const* bboxes_{nullptr};
    VecReal3 const* centers_{nullptr};
    static constexpr size_type candidates_per_axis_{3};

    //// HELPER FUNCTIONS ////

    // Create sorted and uniquified X, Y, Z values of bbox centers
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
