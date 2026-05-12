//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/OrangeTypes.hh"

#include "../OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Access data for a BIH inner node.
 */
class BIHInternalNodeView
{
  public:
    //!@{
    //! \name Type aliases
    using Side = BIHInnerNode::Side;
    //!@}

    // Construct from inner node data
    inline CELER_FUNCTION explicit BIHInternalNodeView(BIHInnerNode const& node);

    // Get partition axis
    inline CELER_FUNCTION Axis axis() const;

    // Get child node for a side
    inline CELER_FUNCTION BIHNodeId child(Side side) const;

    // Get edge bounding box for a side
    inline CELER_FUNCTION FastBBox const& bbox(Side side) const;

    // Get edge bounding plane position for a side
    inline CELER_FUNCTION fast_real_type bounding_plane_pos(Side side) const;

  private:
    BIHInnerNode const& node_;
};

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
    using SpanLocalVol = LdgSpan<LocalVolumeId const>;
    //!@}

    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION
    BIHView(BIHTreeRecord const& tree, Storage const& storage);

    // Determine if a node is inner, i.e., not a leaf
    inline CELER_FUNCTION bool is_inner(BIHNodeId id) const;

    // Get an inner node for a given BIHNodeId
    inline CELER_FUNCTION BIHInternalNodeView inner_node(BIHNodeId id) const;

    // Get the bbox for a given vol_id.
    inline CELER_FUNCTION FastBBox const& bbox(LocalVolumeId vol_id) const;

    // Get the vol_ids on a given leaf node
    inline CELER_FUNCTION SpanLocalVol leaf_vol_ids(BIHNodeId) const;

    // Get the inf_vol_ids
    inline CELER_FUNCTION SpanLocalVol inf_vol_ids() const;

  private:
    //// DATA ////
    BIHTreeRecord const& tree_;
    Storage const& storage_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from an inner node.
 */
CELER_FUNCTION
BIHInternalNodeView::BIHInternalNodeView(BIHInnerNode const& node)
    : node_(node)
{
    CELER_EXPECT(node_);
}

//---------------------------------------------------------------------------//
/*!
 * Get partition axis.
 */
CELER_FUNCTION Axis BIHInternalNodeView::axis() const
{
    return node_.axis;
}

//---------------------------------------------------------------------------//
/*!
 * Get child node for a side.
 */
CELER_FUNCTION BIHNodeId
BIHInternalNodeView::child(BIHInternalNodeView::Side side) const
{
    return node_.edges[side].child;
}

//---------------------------------------------------------------------------//
/*!
 * Get edge bounding box for a side.
 */
CELER_FUNCTION FastBBox const&
BIHInternalNodeView::bbox(BIHInternalNodeView::Side side) const
{
    return node_.edges[side].bbox;
}

//---------------------------------------------------------------------------//
/*!
 * Get edge bounding plane position for a side.
 */
CELER_FUNCTION fast_real_type
BIHInternalNodeView::bounding_plane_pos(BIHInternalNodeView::Side side) const
{
    return node_.edges[side].bounding_plane_pos;
}

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
BIHInternalNodeView BIHView::inner_node(BIHNodeId id) const
{
    CELER_EXPECT(this->is_inner(id));
    return BIHInternalNodeView{
        storage_.inner_nodes[tree_.inner_nodes[id.unchecked_get()]]};
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
CELER_FUNCTION auto BIHView::leaf_vol_ids(BIHNodeId id) const -> SpanLocalVol
{
    CELER_EXPECT(!this->is_inner(id));
    ItemId<BIHLeafNode> leaf_id
        = tree_.leaf_nodes[id.unchecked_get() - tree_.inner_nodes.size()];
    auto const& leaf_node = storage_.leaf_nodes[leaf_id];
    return storage_.local_volume_ids[leaf_node.vol_ids];
}

//---------------------------------------------------------------------------//
/*!
 *  Get the inf_vol_ids.
 */
CELER_FUNCTION auto BIHView::inf_vol_ids() const -> SpanLocalVol
{
    return storage_.local_volume_ids[tree_.inf_vol_ids];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
