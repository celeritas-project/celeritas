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
#include "../inp/Bih.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a bounding interval hierarchy from supplied bounding boxes.
 *
 * This implementation matches the structure proposed in the original paper
 * \citep{wachter-bih-2006, https://doi.org/10.2312/EGWR/EGSR06/139-149}.
 * Construction is done recursively. At a given recursion depth, if at least
 * \param max_split_size bounding boxes are present, partitioning is attempted.
 * Partitioning is done on the basis of bounding box centers using the "longest
 * dimension" heuristic.  If more than \param max_split_size bounding boxes are
 * present, or if the bounding boxes are are non-partitionable (i.e., they all
 * have the same center), a leaf node is created.
 *
 * Any bounding boxes that have at least one infinite dimension are stored in
 * inf_vols. In the event that all bounding boxes are infinite, the tree will
 * consist of a single empty leaf node with all volumes in the stored inf_vols.
 * This final case should not occur unless an ORANGE geometry is created via
 * a method where volume bounding boxes are not available.
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
    using Input = inp::BIHBuilder;
    //!@}

  public:
    // Construct from Storage and Input objects
    BIHBuilder(Storage* storage, Input inp);

    // Create BIH Nodes
    BIHTreeRecord
    operator()(VecBBox&& bboxes, SetLocalVolId const& implicit_vol_id);

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

    Input inp_;

    //// HELPER FUNCTIONS ////

    // Recursively construct BIH nodes for a vector of bbox indices
    void construct_tree(VecIndices const& indices,
                        VecNodes* nodes,
                        BIHNodeId parent,
                        FastBBox const& bbox,
                        size_type current_depth,
                        size_type& depth);

    // Separate nodes into inner and leaf vectors and renumber accordingly
    ArrangedNodes arrange_nodes(VecNodes const& nodes) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
