//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.hh
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
 * Create a bounding interval hierarchy from supplied bounding boxes.
 *
 * This implementation matches the structure proposed in the original
 * paper [1]. Partitioning is done on the basis of bounding box centers using
 * the "longest dimension" heuristic. All leaf nodes contain either a single
 * volume id, or multiple volume ids if the volumes have bounding boxes that
 * share the same center.
 *
 * [1] C. Wachter, Carsten and A. Keller, "Instant Ray Tracing: The Bounding
 * Interval Hierarchy" Eurographics Symposium on Rendering, 2006,
 * doi:10.2312/EGWR/EGSR06/139-149}
 */
class BIHBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using VecBBox = std::vector<BoundingBox>;
    using LVIStorage = Collection<LocalVolumeId,
                                  Ownership::value,
                                  MemSpace::host,
                                  OpaqueId<LocalVolumeId>>;
    using NodeStorage
        = Collection<BIHNode, Ownership::value, MemSpace::host, OpaqueId<BIHNode>>;
    using VecNodes = std::vector<BIHNode>;
    //!@}

  public:
    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    explicit CELER_FUNCTION BIHBuilder(VecBBox bboxes,
                                       LVIStorage* lvi_storage,
                                       NodeStorage* node_storage);

    // Create BIH Nodes
    CELER_FUNCTION BIHParams operator()() const;

  private:
    /// TYPES ///

    using VecReal3 = std::vector<Real3>;
    using VecIndices = std::vector<LocalVolumeId>;
    using PairVecIndices = std::pair<VecIndices, VecIndices>;
    using VecAxes = std::vector<Axis>;
    using AxesCenters = std::vector<std::vector<double>>;

    struct Partition
    {
        Axis axis = Axis::size_;
        real_type location;
        explicit CELER_FUNCTION operator bool() const
        {
            return axis != Axis::size_;
        }
    };

    //// DATA ////

    VecBBox bboxes_;
    LVIStorage* lvi_storage_;
    NodeStorage* node_storage_;

    //// HELPER FUNCTIONS ////

    // Recursively construct BIH nodes for a vector of bbox indices
    void construct_tree(VecIndices const& indices, VecNodes& nodes) const;

    // Find a suitable partition for the given bounding boxes
    Partition
    find_partition(VecIndices const& indicies, VecReal3 const& centers) const;

    // Divide bboxes into left and right branches based on a partition
    PairVecIndices partition_bboxes(VecIndices const& indices,
                                    VecReal3 const& centers,
                                    Partition const& p) const;

    // Add leaf volume ids to a given node
    void make_leaf(BIHNode& node, VecIndices const& indices) const;

    // Calculate the centers of each bounding box
    CELER_FUNCTION VecReal3 centers(VecIndices const& indices) const;

    // Create sorted and uniquified X, Y, Z values of bbox centers
    AxesCenters axes_centers(VecReal3 const& centers) const;

    // Bounding box of a collection of bounding boxes
    CELER_FUNCTION BoundingBox meta_bbox(VecIndices const& indices) const;

    // Create a vector of axes sorted from longest to shortest.
    CELER_FUNCTION VecAxes sort_axes(BoundingBox const& bbox) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
