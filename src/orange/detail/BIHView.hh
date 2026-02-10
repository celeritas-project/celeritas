//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHView.hh
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
class BIHView
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BIHTreeData>;
    //!@}

    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION
    BIHView(BIHTreeRecord const& tree, Storage const& storage);

    // Determine if a node is inner, i.e., not a leaf
    inline CELER_FUNCTION bool is_inner(BIHNodeId id) const;

    // Get an inner node for a given BIHNodeId
    inline CELER_FUNCTION BIHInnerNode const& inner_node(BIHNodeId id) const;

    // Get a leaf node for a given BIHNodeId
    inline CELER_FUNCTION BIHLeafNode const& leaf_node(BIHNodeId id) const;

    // Get the bbox for a given vol_id.
    inline CELER_FUNCTION FastBBox const& bbox(LocalVolumeId vol_id) const;

    // Get the vol_ids on a given leaf node
    inline CELER_FUNCTION Span<LocalVolumeId const>
    leaf_vol_ids(BIHLeafNode const& leaf) const;

    // Get the inf_vol_ids
    inline CELER_FUNCTION Span<LocalVolumeId const> inf_vol_ids() const;

  private:
    //// DATA ////
    BIHTreeRecord const& tree_;
    Storage const& storage_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage.
 */
CELER_FUNCTION
BIHView::BIHView(BIHTreeRecord const& tree, BIHView::Storage const& storage)
    : tree_(tree), storage_(storage)
{
    CELER_EXPECT(tree);
}

//---------------------------------------------------------------------------//
/*!
 *  Determine if a node is inner, i.e., not a leaf.
 */
CELER_FUNCTION
bool BIHView::is_inner(BIHNodeId id) const
{
    return id.unchecked_get() < tree_.inner_nodes.size();
}

//---------------------------------------------------------------------------//
/*!
 *  Get an inner node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHInnerNode const& BIHView::inner_node(BIHNodeId id) const
{
    CELER_EXPECT(this->is_inner(id));
    return storage_.inner_nodes[tree_.inner_nodes[id.unchecked_get()]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get a leaf node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHLeafNode const& BIHView::leaf_node(BIHNodeId id) const
{
    CELER_EXPECT(!this->is_inner(id));
    return storage_.leaf_nodes[tree_.leaf_nodes[id.unchecked_get()
                                                - tree_.inner_nodes.size()]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get the bbox for a given vol_id.
 */
CELER_FUNCTION FastBBox const& BIHView::bbox(LocalVolumeId vol_id) const
{
    CELER_EXPECT(vol_id.unchecked_get() < tree_.bboxes.size());
    return storage_.bboxes[tree_.bboxes[vol_id]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get the vol_ids on a given leaf node.
 */
CELER_FUNCTION Span<LocalVolumeId const>
BIHView::leaf_vol_ids(BIHLeafNode const& leaf) const
{
    return storage_.local_volume_ids[leaf.vol_ids];
}

//---------------------------------------------------------------------------//
/*!
 *  Get the inf_vol_ids.
 */
CELER_FUNCTION Span<LocalVolumeId const> BIHView::inf_vol_ids() const
{
    return storage_.local_volume_ids[tree_.inf_vol_ids];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
