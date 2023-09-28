//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>
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
 * share the same center. A tree may consist of a single leaf node if the
 * tree contains only 1 volume, or multiple non-partitionable volumes. In the
 * event that all bounding boxes are infinite, the tree will consist of a
 * single empty leaf node with all volumes in the stored inf_vols. This final
 * case is useful in the event that an ORANGE geometry is created via a method
 * where volume bounding boxes are not availible.
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
    using VecBBox = std::vector<FastBBox>;
    using Storage = BIHTreeData<Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Construct from a Storage object
    explicit BIHBuilder(Storage* storage);

    // Create BIH Nodes
    BIHTree operator()(VecBBox bboxes);

  private:
    /// TYPES ///

    using Real3 = Array<fast_real_type, 3>;
    using VecReal3 = std::vector<Real3>;
    using VecIndices = std::vector<LocalVolumeId>;
    using PairVecIndices = std::pair<VecIndices, VecIndices>;
    using AxesCenters = std::vector<std::vector<real_type>>;
    using VecNodes = std::vector<std::variant<BIHInnerNode, BIHLeafNode>>;
    using VecInnerNodes = std::vector<BIHInnerNode>;
    using VecLeafNodes = std::vector<BIHLeafNode>;
    using ArrangedNodes = std::pair<VecInnerNodes, VecLeafNodes>;

    //// DATA ////

    VecBBox bboxes_;
    VecReal3 centers_;
    Storage* storage_;

    //// HELPER FUNCTIONS ////

    // Recursively construct BIH nodes for a vector of bbox indices
    void construct_tree(VecIndices const& indices,
                        VecNodes* nodes,
                        BIHNodeId parent) const;

    // Seperate nodes into inner and leaf vectors and renumber accordingly
    ArrangedNodes arrange_nodes(VecNodes nodes) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
