//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHEnclosingVolFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "BIHView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse the BIH tree to find a volume that contains a point.
 *
 * Traversal is carried out using a depth-first search and terminated as soon
 * as a suitable volume is found. When visiting a leaf node, the bounding boxes
 * of leaf volumes are tested (inexpensive) before testing the leaf volumes
 * themselves (expensive). Determining if the point is enclosed by the volume
 * itself is done with a supplied predicate, which can also be used to exclude
 * volumes based on more stringent criteria. For example, for surface crossing
 * operations, a predicate that excludes the volume a particle is in prior to
 * the crossing may be used.
 *
 * \todo move to top-level orange directory out of detail namespace
 */
class BIHEnclosingVolFinder
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BIHTreeData>;
    //!@}

    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION
    BIHEnclosingVolFinder(BIHTree const& tree, Storage const& storage);

    // Find a volume that satisfies is_inside
    template<class F>
    inline CELER_FUNCTION LocalVolumeId operator()(Real3 const& pos,
                                                   F&& is_inside) const;

  private:
    //// DATA ////
    BIHView view_;

    //// HELPER FUNCTIONS ////

    // Get the ID of the next node in the traversal sequence
    inline CELER_FUNCTION BIHNodeId next_node(BIHNodeId current_id,
                                              BIHNodeId previous_id,
                                              Real3 const& pos) const;

    // Determine if traversal shall proceed down a given edge
    inline CELER_FUNCTION bool visit_edge(BIHInnerNode const& node,
                                          BIHInnerNode::Side side,
                                          Real3 const& pos) const;

    // Determine if any leaf node volumes contain the point
    template<class F>
    inline CELER_FUNCTION LocalVolumeId visit_leaf(BIHLeafNode const& leaf_node,
                                                   Real3 const& pos,
                                                   F&& is_inside) const;

    // Determine if any inf_vols contain the point
    template<class F>
    inline CELER_FUNCTION LocalVolumeId visit_inf_vols(F&& is_inside) const;

    // Determine if a single bbox contains the point
    inline CELER_FUNCTION bool
    visit_bbox(LocalVolumeId const& id, Real3 const& pos) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage.
 */
CELER_FUNCTION
BIHEnclosingVolFinder::BIHEnclosingVolFinder(BIHTree const& tree,
                                             Storage const& storage)
    : view_(tree, storage)
{
}

//---------------------------------------------------------------------------//
/*!
 * Find a volume that satisfies is_inside.
 */
template<class F>
CELER_FUNCTION LocalVolumeId
BIHEnclosingVolFinder::operator()(Real3 const& pos, F&& is_inside) const
{
    BIHNodeId previous_node;
    BIHNodeId current_node{0};
    LocalVolumeId id;

    // Depth-first search
    do
    {
        if (!view_.is_inner(current_node))
        {
            id = this->visit_leaf(
                view_.leaf_node(current_node), pos, is_inside);

            if (id)
            {
                return id;
            }
        }

        previous_node = exchange(
            current_node, this->next_node(current_node, previous_node, pos));

    } while (current_node);

    return this->visit_inf_vols(is_inside);
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 *  Get the ID of the next node in the traversal sequence.
 */
CELER_FUNCTION
BIHNodeId BIHEnclosingVolFinder::next_node(BIHNodeId current_id,
                                           BIHNodeId previous_id,
                                           Real3 const& pos) const
{
    using Side = BIHInnerNode::Side;

    BIHNodeId next_id;

    if (view_.is_inner(current_id))
    {
        auto const& current_node = view_.inner_node(current_id);
        if (previous_id == current_node.parent)
        {
            // Visiting this inner node for the first time; go down either left
            // or right edge
            if (this->visit_edge(current_node, Side::left, pos))
            {
                next_id = current_node.edges[Side::left].child;
            }
            else
            {
                next_id = current_node.edges[Side::right].child;
            }
        }
        else if (previous_id == current_node.edges[Side::left].child)
        {
            // Visiting this inner node for the second time; go down right edge
            // or return to parent
            if (this->visit_edge(current_node, Side::right, pos))
            {
                next_id = current_node.edges[Side::right].child;
            }
            else
            {
                next_id = current_node.parent;
            }
        }
        else
        {
            // Visiting this inner node for the third time; return to parent
            CELER_EXPECT(previous_id == current_node.edges[Side::right].child);
            next_id = current_node.parent;
        }
    }
    else
    {
        // Leaf node; return to parent
        CELER_EXPECT(previous_id == view_.leaf_node(current_id).parent);
        next_id = previous_id;
    }

    return next_id;
}

//---------------------------------------------------------------------------//
/*!
 * Determine if traversal shall proceed down a given edge.
 */
CELER_FUNCTION
bool BIHEnclosingVolFinder::visit_edge(BIHInnerNode const& node,
                                       BIHInnerNode::Side side,
                                       Real3 const& pos) const
{
    CELER_EXPECT(side < BIHInnerNode::Side::size_);

    auto bp_pos = node.edges[side].bounding_plane_pos;
    auto p = pos[to_int(node.axis)];

    return (side == BIHInnerNode::Side::left) ? (p < bp_pos) : (bp_pos < p);
}

//---------------------------------------------------------------------------//
/*!
 * Determine if any leaf node volumes contain the point.
 */
template<class F>
CELER_FUNCTION LocalVolumeId BIHEnclosingVolFinder::visit_leaf(
    BIHLeafNode const& leaf_node, Real3 const& pos, F&& is_inside) const
{
    for (auto id : view_.leaf_vol_ids(leaf_node))
    {
        if (this->visit_bbox(id, pos) && is_inside(id))
        {
            return id;
        }
    }
    return LocalVolumeId{};
}

//---------------------------------------------------------------------------//
/*!
 * Determine if any volumes in inf_vols contain the point.
 */
template<class F>
CELER_FUNCTION LocalVolumeId
BIHEnclosingVolFinder::visit_inf_vols(F&& is_inside) const
{
    for (auto id : view_.inf_vol_ids())
    {
        if (is_inside(id))
        {
            return id;
        }
    }
    return LocalVolumeId{};
}

//---------------------------------------------------------------------------//
/*!
 * Determinate if a single bbox contains the point.
 */
CELER_FUNCTION
bool BIHEnclosingVolFinder::visit_bbox(LocalVolumeId const& id,
                                       Real3 const& point) const
{
    return is_inside(view_.bbox(id), point);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
