//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>
#include <variant>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "geocel/BoundingBox.hh"

#include "BIHPartitioner.hh"
#include "../OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a bounding interval hierarchy from supplied bounding boxes.
 *
 * This implementation matches the structure proposed in the original
 * paper \citep{wachter-bih-2006, https://doi.org/10.2312/EGWR/EGSR06/139-149}.
 * Partitioning is done on the basis of bounding box centers using the "longest
 * dimension" heuristic. All leaf nodes contain either a single volume id, or
 * multiple volume ids if the volumes have bounding boxes that share the same
 * center. A tree may consist of a single leaf node if the tree contains only 1
 * volume, or multiple non-partitionable volumes. In the event that all
 * bounding boxes are infinite, the tree will consist of a single empty leaf
 * node with all volumes in the stored inf_vols. This final case is useful in
 * the event that an ORANGE geometry is created via a method where volume
 * bounding boxes are not available.
 *
 * Bounding boxes supplied to this builder should "bumped," i.e. expanded
 * outward by at least floating-point epsilson from the volumes they bound.
 * This eliminates the possibility of accidentally missing a volume during
 * tracking.
 */
class BIHBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using VecBBox = std::vector<FastBBox>;
    using Storage = BIHTreeData<Ownership::value, MemSpace::host>;
    using SetLocalVolId = std::set<LocalVolumeId>;
    //!@}

  public:
    // Construct from a Storage object
    explicit BIHBuilder(Storage* storage);

    // Create BIH Nodes
    BIHTree operator()(VecBBox&& bboxes, SetLocalVolId const& implicit_vol_ids);

  private:
    /// TYPES ///

    using Real3 = Array<fast_real_type, 3>;
    using VecIndices = std::vector<LocalVolumeId>;
    using VecNodes = std::vector<std::variant<BIHInnerNode, BIHLeafNode>>;
    using VecInnerNodes = std::vector<BIHInnerNode>;
    using VecLeafNodes = std::vector<BIHLeafNode>;
    using ArrangedNodes = std::pair<VecInnerNodes, VecLeafNodes>;

    struct Temporaries
    {
        VecBBox bboxes;
        std::vector<Real3> centers;
    };

    //// DATA ////

    Temporaries temp_;

    CollectionBuilder<FastBBox> bboxes_;
    CollectionBuilder<LocalVolumeId> local_volume_ids_;
    CollectionBuilder<BIHInnerNode> inner_nodes_;
    CollectionBuilder<BIHLeafNode> leaf_nodes_;

    //// HELPER FUNCTIONS ////

    // Recursively construct BIH nodes for a vector of bbox indices
    void construct_tree(VecIndices const& indices,
                        VecNodes* nodes,
                        BIHNodeId parent,
                        FastBBox const& bbox);

    // Separate nodes into inner and leaf vectors and renumber accordingly
    ArrangedNodes arrange_nodes(VecNodes const& nodes) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
