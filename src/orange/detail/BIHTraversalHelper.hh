//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHTraversal.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse BIH tree using a depth-first search.
 *
 * \todo move to top-level orange directory out of detail namespace
 */
class BIHTraversalHelper
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BIHTreeData>;
    //!@}

    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION
    BIHTraversalHelper(BIHTree const& tree, Storage const& storage);

    // Determine if a node is inner, i.e., not a leaf
    inline CELER_FUNCTION bool is_inner(BIHNodeId id) const;

    // Get an inner node for a given BIHNodeId
    inline CELER_FUNCTION BIHInnerNode const&
    get_inner_node(BIHNodeId id) const;

    // Get a leaf node for a given BIHNodeId
    inline CELER_FUNCTION BIHLeafNode const& get_leaf_node(BIHNodeId id) const;

    // Get a bbox for a given vol_id
    inline CELER_FUNCTION FastBBox const& get_bbox(LocalVolumeId vol_id) const;

    // Get the vol_id of the ith volume on a given leaf node
    inline CELER_FUNCTION LocalVolumeId get_leaf_volid(BIHLeafNode leaf,
                                                       size_type i) const;

    // Get the number of inf_volids on the tree
    inline CELER_FUNCTION size_type get_num_inf_volids() const;

    // Get the ith vol_id in inf_volids
    inline CELER_FUNCTION LocalVolumeId get_inf_volid(size_type i) const;

  private:
    //// DATA ////
    BIHTree const& tree_;
    Storage const& storage_;
    size_type leaf_offset_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage.
 */
CELER_FUNCTION
BIHTraversalHelper::BIHTraversalHelper(
    BIHTree const& tree, BIHTraversalHelper::Storage const& storage)
    : tree_(tree), storage_(storage), leaf_offset_(tree.inner_nodes.size())
{
    CELER_EXPECT(tree);
}

//---------------------------------------------------------------------------//
/*!
 *  Determine if a node is inner, i.e., not a leaf.
 */
CELER_FUNCTION
bool BIHTraversalHelper::is_inner(BIHNodeId id) const
{
    return id.unchecked_get() < leaf_offset_;
}

//---------------------------------------------------------------------------//
/*!
 *  Get an inner node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHInnerNode const& BIHTraversalHelper::get_inner_node(BIHNodeId id) const
{
    CELER_EXPECT(this->is_inner(id));
    return storage_.inner_nodes[tree_.inner_nodes[id.unchecked_get()]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get a leaf node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHLeafNode const& BIHTraversalHelper::get_leaf_node(BIHNodeId id) const
{
    CELER_EXPECT(!this->is_inner(id));
    return storage_
        .leaf_nodes[tree_.leaf_nodes[id.unchecked_get() - leaf_offset_]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get a leaf node for a given BIHNodeId.
 */
CELER_FUNCTION FastBBox const&
BIHTraversalHelper::get_bbox(LocalVolumeId vol_id) const
{
    CELER_EXPECT(vol_id.unchecked_get() < tree_.bboxes.size());
    return storage_.bboxes[tree_.bboxes[vol_id]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get the vol_id of the ith volume on a given leaf node.
 */
CELER_FUNCTION LocalVolumeId
BIHTraversalHelper::get_leaf_volid(BIHLeafNode leaf, size_type i) const
{
    CELER_EXPECT(i < leaf.vol_ids.size());
    return storage_.local_volume_ids[leaf.vol_ids[i]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get the number of inf_volids on the tree.
 */
CELER_FUNCTION size_type BIHTraversalHelper::get_num_inf_volids() const
{
    return tree_.inf_volids.size();
}

//---------------------------------------------------------------------------//
/*!
 *  Get the ith vol_id in inf_volids.
 */
CELER_FUNCTION LocalVolumeId BIHTraversalHelper::get_inf_volid(size_type i) const
{
    CELER_EXPECT(i < tree_.inf_volids.size());
    return storage_.local_volume_ids[tree_.inf_volids[i]];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
