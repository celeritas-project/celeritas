//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BvhSafetyCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/cont/IdStack.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

#include "BvhView.hh"
#include "../BoundingBoxUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse the BVH to bound the distance from a point to the nearest volume.
 *
 * The result is a lower bound on the distance from a point that is \em outside
 * every volume in the tree to the nearest of those volumes, limited to a
 * maximum search distance. Because a volume is enclosed by its bounding box,
 * the distance \f$ d_b \f$ from the point to the box cannot exceed the
 * distance to the volume. A caller-provided functor gives a second lower
 * bound \f$ d_v \f$ for a single volume (or zero if it has none), and
 * the larger of the two is used:
 * \f[
   d = \min\left(d_\mathrm{max}, \min_v \max(d_{b,v}, d_v) \right) \,.
 * \f]
 *
 * Traversal is a depth-first search, nearest child first, that skips any
 * subtree or volume whose bounding box is no closer than the current result.
 * Volumes without a finite bounding box have no \f$ d_b \f$ and are always
 * evaluated. The search stops as soon as the result reaches zero.
 */
class BvhSafetyCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BvhTreeData>;
    //!@}

  public:
    // Construct from a tree record and storage
    inline CELER_FUNCTION BvhSafetyCalculator(BvhTreeRecord const& tree,
                                              Storage const& storage);

    // Calculate a lower bound on the distance to the nearest volume
    template<class F>
    inline CELER_FUNCTION real_type operator()(
        Real3 const& pos, F&& calc_vol_safety, real_type max_dist) const;

  private:
    //// TYPES ////

    struct Child
    {
        BvhNodeId id;
        real_type dist_sq;
    };

    //// DATA ////

    BvhView view_;

    //// HELPER FUNCTIONS ////

    // Update the result with the volumes on a leaf
    template<class F>
    inline CELER_FUNCTION real_type visit_leaf(BvhNodeId leaf_id,
                                               Real3 const& pos,
                                               real_type result,
                                               F&& calc_vol_safety) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from a tree record and storage.
 */
CELER_FUNCTION BvhSafetyCalculator::BvhSafetyCalculator(
    BvhTreeRecord const& tree, Storage const& storage)
    : view_(tree, storage)
{
}

//---------------------------------------------------------------------------//
/*!
 * Calculate a lower bound on the distance to the nearest volume.
 *
 * The \c calc_vol_safety argument should have the signature
 * \code real_type(LocalVolumeId) \endcode
 * and return a lower bound on the distance from the point to the boundary of
 * the given volume, or zero if no bound is available.
 */
template<class F>
CELER_FUNCTION real_type BvhSafetyCalculator::operator()(
    Real3 const& pos, F&& calc_vol_safety, real_type max_dist) const
{
    CELER_EXPECT(max_dist > 0);

    using Side = BvhInternalNode::Side;
    using StackT = IdStack<BvhNodeId, max_bvh_depth - 1>;

    real_type result = max_dist;

    // Volumes without finite bounding boxes cannot be pruned
    for (LocalVolumeId id : view_.inf_vol_ids())
    {
        result = celeritas::min(result, calc_vol_safety(id));
        if (result == 0)
        {
            return 0;
        }
    }

    // Stack of deferred nodes
    BvhNodeId stack_storage[StackT::spill_extent];
    StackT stack{stack_storage};
    stack.push(BvhNodeId{0});

    while (!stack.empty() && result > 0)
    {
        BvhNodeId const node_id = stack.top();
        stack.pop();

        if (!view_.is_internal(node_id))
        {
            result = this->visit_leaf(node_id, pos, result, calc_vol_safety);
            continue;
        }

        auto const& node = view_.inner_node(node_id);

        Child near{node.child(Side::left),
                   calc_squared_distance(node.bbox(Side::left), pos)};
        Child far{node.child(Side::right),
                  calc_squared_distance(node.bbox(Side::right), pos)};

        if (far.dist_sq < near.dist_sq)
        {
            trivial_swap(near, far);
        }

        // Prune both children if even the nearer one is too far away
        real_type const result_sq = ipow<2>(result);
        if (near.dist_sq >= result_sq)
        {
            continue;
        }

        // Defer the farther child so the nearer one is visited first
        if (far.dist_sq < result_sq)
        {
            stack.push(far.id);
        }
        stack.push(near.id);
    }

    CELER_ENSURE(result >= 0 && result <= max_dist);
    return result;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Update the result with the volumes on a leaf.
 */
template<class F>
CELER_FUNCTION real_type BvhSafetyCalculator::visit_leaf(
    BvhNodeId leaf_id,
    Real3 const& pos,
    real_type result,
    F&& calc_vol_safety) const
{
    for (LocalVolumeId id : view_.leaf_vol_ids(leaf_id))
    {
        real_type const bbox_sq = calc_squared_distance(view_.bbox(id), pos);
        if (bbox_sq >= ipow<2>(result))
        {
            continue;
        }

        // Both the bounding box distance and the volume's own estimate are
        // lower bounds, so use the tighter (larger) one
        result = celeritas::min(
            result, celeritas::max(std::sqrt(bbox_sq), calc_vol_safety(id)));
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
