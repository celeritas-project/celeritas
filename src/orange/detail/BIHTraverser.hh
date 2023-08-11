//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHTraverser.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"
#include "orange/BoundingBoxUtils.hh"
#include "orange/OrangeData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse BIH tree using a depth-first search.
 */
class BIHTraverser
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BIHTreeData>;
    //!@}

    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    explicit inline CELER_FUNCTION
    BIHTraverser(BIHTree const& tree, Storage const& storage);

    // Point-in-volume operation
    template<class F>
    inline CELER_FUNCTION LocalVolumeId operator()(Real3 const& point,
                                                   F&& is_inside) const;

  private:
    //// DATA ////
    BIHTree const& tree_;
    Storage const& storage_;
    size_type leaf_offset_;

    //// HELPER FUNCTIONS ////

    // Get the ID of the next node in the traversal sequence
    inline CELER_FUNCTION BIHNodeId next_node(BIHNodeId const& current_id,
                                              BIHNodeId const& previous_id,
                                              Real3 const& point) const;

    // Determine if traversal shall proceed down a given edge
    inline CELER_FUNCTION bool visit_edge(BIHInnerNode const& node,
                                          BIHInnerNode::Edge edge,
                                          Real3 const& point) const;

    // Determine if a node is inner, i.e., not a leaf
    inline CELER_FUNCTION bool is_inner(BIHNodeId id) const;

    // Get an inner node for a given BIHNodeId
    inline CELER_FUNCTION BIHInnerNode const&
    get_inner_node(BIHNodeId id) const;

    // Get a leaf node for a given BIHNodeId
    inline CELER_FUNCTION BIHLeafNode const& get_leaf_node(BIHNodeId id) const;

    // Determine if any leaf node volumes contain the point
    template<class F>
    inline CELER_FUNCTION LocalVolumeId visit_leaf(BIHLeafNode const& leaf_node,
                                                   Real3 const& point,
                                                   F&& is_inside) const;

    // Determine if any inf_vols contain the point
    template<class F>
    inline CELER_FUNCTION LocalVolumeId visit_inf_vols(F&& is_inside) const;

    // Determine if a single bbox contains the point
    inline CELER_FUNCTION bool
    visit_bbox(LocalVolumeId const& id, Real3 const& point) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage.
 */
CELER_FUNCTION
BIHTraverser::BIHTraverser(BIHTree const& tree,
                           BIHTraverser::Storage const& storage)
    : tree_(tree), storage_(storage), leaf_offset_(tree.inner_nodes.size())
{
    CELER_EXPECT(tree);
}

//---------------------------------------------------------------------------//
/*!
 * Point-in-volume operation.
 */
template<class F>
CELER_FUNCTION LocalVolumeId BIHTraverser::operator()(Real3 const& point,
                                                      F&& is_inside) const
{
    BIHNodeId previous_node;
    BIHNodeId current_node{0};
    LocalVolumeId id;

    do
    {
        if (!this->is_inner(current_node))
        {
            id = this->visit_leaf(
                this->get_leaf_node(current_node), point, is_inside);

            if (id)
            {
                return id;
            }
        }

        previous_node = exchange(
            current_node, this->next_node(current_node, previous_node, point));

    } while (current_node);

    if (!id)
    {
        id = this->visit_inf_vols(is_inside);
    }

    return id;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 *  Get the ID of the next node in the traversal sequence.
 */
CELER_FUNCTION
BIHNodeId BIHTraverser::next_node(BIHNodeId const& current_id,
                                  BIHNodeId const& previous_id,
                                  Real3 const& point) const
{
    using Edge = BIHInnerNode::Edge;

    BIHNodeId next_id;

    if (this->is_inner(current_id))
    {
        auto const& current_node = this->get_inner_node(current_id);
        if (previous_id == current_node.parent)
        {
            // Visiting this inner node for the first time; go down either left
            // or right edge
            if (this->visit_edge(current_node, Edge::left, point))
            {
                next_id = current_node.bounding_planes[Edge::left].child;
            }
            else
            {
                next_id = current_node.bounding_planes[Edge::right].child;
            }
        }
        else if (previous_id == current_node.bounding_planes[Edge::left].child)
        {
            // Visiting this inner node for the second time; go down right edge
            // or return to parent
            if (this->visit_edge(current_node, Edge::right, point))
            {
                next_id = current_node.bounding_planes[Edge::right].child;
            }
            else
            {
                next_id = current_node.parent;
            }
        }
        else
        {
            // Visiting this inner node for the third time; return to parent
            CELER_EXPECT(previous_id
                         == current_node.bounding_planes[Edge::right].child);
            next_id = current_node.parent;
        }
    }
    else
    {
        // Leaf node; return to parent
        CELER_EXPECT(previous_id == this->get_leaf_node(current_id).parent);
        next_id = previous_id;
    }

    return next_id;
}

//---------------------------------------------------------------------------//
/*!
 * Determine if traversal shall proceed down a given edge.
 */
CELER_FUNCTION
bool BIHTraverser::visit_edge(BIHInnerNode const& node,
                              BIHInnerNode::Edge edge,
                              Real3 const& point) const
{
    CELER_EXPECT(edge < BIHInnerNode::Edge::size_);

    auto pos = node.bounding_planes[edge].position;
    auto point_pos = point[to_int(node.axis)];

    return (edge == BIHInnerNode::Edge::left) ? (point_pos < pos)
                                              : (pos < point_pos);
}

//---------------------------------------------------------------------------//
/*!
 *  Determine if a node is inner, i.e., not a leaf.
 */
CELER_FUNCTION
bool BIHTraverser::is_inner(BIHNodeId id) const
{
    return id.unchecked_get() < leaf_offset_;
}

//---------------------------------------------------------------------------//
/*!
 *  Get an inner node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHInnerNode const& BIHTraverser::get_inner_node(BIHNodeId id) const
{
    CELER_EXPECT(this->is_inner(id));
    return storage_.inner_nodes[tree_.inner_nodes[id.unchecked_get()]];
}

//---------------------------------------------------------------------------//
/*!
 *  Get a leaf node for a given BIHNodeId.
 */
CELER_FUNCTION
BIHLeafNode const& BIHTraverser::get_leaf_node(BIHNodeId id) const
{
    CELER_EXPECT(!this->is_inner(id));
    return storage_
        .leaf_nodes[tree_.leaf_nodes[id.unchecked_get() - leaf_offset_]];
}

//---------------------------------------------------------------------------//
/*!
 * Determine if any leaf node volumes contain the point.
 */
template<class F>
CELER_FUNCTION LocalVolumeId BIHTraverser::visit_leaf(
    BIHLeafNode const& leaf_node, Real3 const& point, F&& is_inside) const
{
    for (auto i : range(leaf_node.vol_ids.size()))
    {
        auto id = storage_.local_volume_ids[leaf_node.vol_ids[i]];
        if (this->visit_bbox(id, point) && is_inside(id))
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
CELER_FUNCTION LocalVolumeId BIHTraverser::visit_inf_vols(F&& is_inside) const
{
    for (auto i : range(tree_.inf_volids.size()))
    {
        auto id = storage_.local_volume_ids[tree_.inf_volids[i]];
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
bool BIHTraverser::visit_bbox(LocalVolumeId const& id, Real3 const& point) const
{
    return is_inside(storage_.bboxes[tree_.bboxes[id]], point);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
