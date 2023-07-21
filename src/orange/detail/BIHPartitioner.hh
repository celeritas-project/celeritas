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
        real_type location;
        explicit CELER_FUNCTION operator bool() const
        {
            return axis != Axis::size_;
        }
    };
    //!@}

  public:
    //! Default constructor
    CELER_FORCEINLINE_FUNCTION BIHPartitioner() {}

    // Construct from vector of bounding boxes and respective centers.
    explicit CELER_FUNCTION BIHPartitioner(VecBBox* bboxes, VecReal3* centers);

    // Find a suitable partition for the given bounding boxes
    CELER_FUNCTION Partition operator()(VecIndices const& indicies) const;

  private:
    /// TYPES ///
    using AxesCenters = std::vector<std::vector<double>>;

    //// DATA ////
    VecBBox* bboxes_;
    VecReal3* centers_;

    //// HELPER FUNCTIONS ////

    // Create sorted and uniquified X, Y, Z values of bbox centers
    AxesCenters axes_centers(VecIndices const& indices) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
