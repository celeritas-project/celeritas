//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHEnclosingVolFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/IdStack.hh"
#include "orange/OrangeTypes.hh"

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
    inline CELER_FUNCTION LocalVolumeId visit_leaf(BIHNodeId leaf_id,
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
    using Side = BIHInternalNode::Side;

    // Stack of deferred nodes
    using StackT = IdStack<BIHNodeId, max_bih_depth - 1>;
    BIHNodeId stack_spill_[StackT::spill_extent];
    StackT stack{stack_spill_};
    stack.push(BIHNodeId{0});

    while (!stack.empty())
    {
        if (!view_.is_internal(stack.top()))
        {
            auto id = this->visit_leaf(stack.top(), pos, is_inside_vol);
            stack.pop();

            if (id)
            {
                return id;
            }
        }
        else
        {
            auto const& node = view_.inner_node(stack.top());
            stack.pop();
            for (auto s : {Side::right, Side::left})
            {
                if (is_inside(node.bbox(s), pos))
                {
                    stack.push(node.child(s));
                }
            }
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
    BIHNodeId leaf_id, Real3 const& pos, F&& is_inside) const
{
    for (auto id : view_.leaf_vol_ids(leaf_id))
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
