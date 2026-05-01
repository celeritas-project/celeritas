//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHEnclosingVolFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "BIHView.hh"
#include "../inp/Bih.hh"

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
    BIHEnclosingVolFinder(BIHTreeRecord const& tree, Storage const& storage);

    // Find a volume that satisfies is_inside
    template<class F>
    inline CELER_FUNCTION LocalVolumeId operator()(Real3 const& pos,
                                                   F&& is_inside) const;

  private:
    //// DATA ////
    BIHView view_;

    //// HELPER FUNCTIONS ////

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
BIHEnclosingVolFinder::BIHEnclosingVolFinder(BIHTreeRecord const& tree,
                                             Storage const& storage)
    : view_(tree, storage)
{
}

//---------------------------------------------------------------------------//
/*!
 * Find a volume that satisfies is_inside_vol.
 */
template<class F>
CELER_FUNCTION LocalVolumeId
BIHEnclosingVolFinder::operator()(Real3 const& pos, F&& is_inside_vol) const
{
    using Side = BIHInnerNode::Side;

    constexpr auto stack_size = inp::BIHBuilder::max_depth_limit - 1;
    BIHNodeId stack[stack_size];
    BIHNodeId::index_type stack_ptr = 0;
    BIHNodeId current_node{0};

    while (current_node)
    {
        if (!view_.is_inner(current_node))
        {
            auto id = this->visit_leaf(
                view_.leaf_node(current_node), pos, is_inside_vol);

            if (id)
            {
                return id;
            }

            CELER_ASSERT(stack_ptr < stack_size);
            current_node = stack_ptr > 0 ? stack[--stack_ptr] : BIHNodeId{};
            continue;
        }

        auto const& node = view_.inner_node(current_node);

        bool in_left = is_inside(node.edges[Side::left].bbox, pos);
        bool in_right = is_inside(node.edges[Side::right].bbox, pos);

        if (in_left && in_right)
        {
            CELER_ASSERT(stack_ptr < stack_size);
            stack[stack_ptr++] = node.edges[Side::right].child;
            current_node = node.edges[Side::left].child;
        }
        else if (in_left)
        {
            current_node = node.edges[Side::left].child;
        }
        else if (in_right)
        {
            current_node = node.edges[Side::right].child;
        }
        else
        {
            CELER_ASSERT(stack_ptr < stack_size);
            current_node = stack_ptr > 0 ? stack[--stack_ptr] : BIHNodeId{};
        }
    }

    return this->visit_inf_vols(is_inside_vol);
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
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
