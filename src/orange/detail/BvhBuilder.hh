//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BvhBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "geocel/BoundingBox.hh"  // IWYU pragma: keep
#include "orange/OrangeTypes.hh"
#include "orange/detail/BvhData.hh"

#include "../inp/Bvh.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create a bounding volume hierarchy (BVH) from the supplied bounding boxes.
 *
 * The class builds a Bounting Volume Hierarchy \citet{ericson-collision-2004,
 * https://www.taylorfrancis.com/books/9780080474144} ) using a top-down,
 * recursive construction method. Leaf nodes are created when one of the
 * following criteria are met:
 *
 * 1) the number of remaining bounding boxes is max_leaf_size or fewer,
 * 2) the remaining bounding boxes are non-partitionable (i.e., they all have
 *    the same center),
 * 3) the current recursion depth has reached the `depth_limit`.
 *
 * Any bounding boxes that have at least one infinite dimension are not stored
 * on the tree, but rather a separate `inf_vols` structure. In the event that
 * all bounding boxes are infinite, the tree will consist of a single empty
 * leaf node with all volumes in the stored `inf_vols`. This final case should
 * not occur unless an ORANGE geometry is created via a method where volume
 * bounding boxes are not available.
 *
 * Bounding boxes supplied to this builder should "bumped," i.e. expanded
 * outward by at least floating-point epsilson from the volumes they bound.
 * This eliminates the possibility of accidentally missing a volume during
 * tracking.
 */
class BvhBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using VecBBox = std::vector<FastBBox>;
    using Storage = BvhTreeData<Ownership::value, MemSpace::host>;
    using SetLocalVolId = std::set<LocalVolumeId>;
    using Input = inp::BvhBuilder;
    //!@}

  public:
    // Construct from Storage and Input objects
    BvhBuilder(Storage* storage, Input inp);

    // Create BVH Nodes
    BvhTreeRecord
    operator()(VecBBox&& bboxes, SetLocalVolId const& implicit_vol_id);

  private:
    /// TYPES ///

    using Real3 = Array<fast_real_type, 3>;
    using VecIndices = std::vector<LocalVolumeId>;
    using VecNodes = std::vector<std::variant<BvhInternalNode, BvhLeafNode>>;
    using VecInnerNodes = std::vector<BvhInternalNode>;
    using VecLeafNodes = std::vector<BvhLeafNode>;
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
    CollectionBuilder<BvhInternalNode> internal_nodes_;
    CollectionBuilder<BvhLeafNode> leaf_nodes_;

    Input inp_;

    //// HELPER FUNCTIONS ////

    // Recursively construct BVH nodes for a vector of bbox indices
    void construct_tree(VecIndices const& indices,
                        VecNodes* nodes,
                        size_type current_depth,
                        size_type& depth);

    // Separate nodes into inner and leaf vectors and renumber accordingly
    ArrangedNodes arrange_nodes(VecNodes const& nodes) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
