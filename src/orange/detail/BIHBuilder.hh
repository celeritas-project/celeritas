//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/cont/Range.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeData.hh"
#include "orange/detail/BIHPartitioner.hh"

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
    using AxesCenters = std::vector<std::vector<double>>;

    //// DATA ////

    VecBBox bboxes_;
    VecReal3 centers_;
    LVIStorage* lvi_storage_;
    NodeStorage* node_storage_;
    BIHPartitioner partitioner_;

    //// HELPER FUNCTIONS ////

    // Recursively construct BIH nodes for a vector of bbox indices
    void construct_tree(VecIndices const& indices,
                        VecNodes& nodes,
                        BIHNodeId parent) const;

    // Add leaf volume ids to a given node
    void make_leaf(BIHNode& node, VecIndices const& indices) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
