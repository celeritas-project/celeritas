//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BvhIntersectingVolFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/IdStack.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

#include "BvhView.hh"
#include "../BoundingBoxUtils.hh"
#include "../univ/detail/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse the BVH to the find the volume that the ray intersects with first.
 *
 * Traversal is carried out using a depth first search. During traversal, the
 * minimum intersection is stored. The decision to traverse an edge is done by
 * calculating the distance to intersection with the precomputed edge bounding
 * box. The edge bounding box is the bounding box created by clipping an
 * infinite bounding box with all bounding planes between the root node and the
 * current edge (inclusive). If a ray's intersection with the edge bbox is
 * found to be nearer than the current minimum intersection, traversal proceeds
 * down that edge. The minimum intersection is modified when a nearer
 * minimumium intersection with a actual volume if found (NOT a nearer
 * intersection with an edge bbox).
 *
 * \todo move to top-level orange directory out of detail namespace
 */
class BvhIntersectingVolFinder
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BvhTreeData>;

    struct Ray
    {
        Real3 const& pos;
        Real3 const& dir;
    };
    //!@}

    // Construct from a vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION BvhIntersectingVolFinder(BvhTreeRecord const& tree,
                                                   Storage const& storage);

    // Calculate the minimum intersection, with supplied maximum search
    // distance
    template<class F>
    inline CELER_FUNCTION Intersection operator()(
        Ray ray, F&& visit_vol, real_type max_search_dist) const;

  private:
    //// DATA ////
    BvhView view_;

    //// HELPER FUNCTIONS ////

    // Determine if the intersection with an edge/vol bbox is less than
    // min_dist
    inline CELER_FUNCTION bool
    visit_bbox(FastBBox const& bbox, Ray ray, real_type min_dist) const;

    // Calculate the current min intersection, which may/may not be on this
    // leaf
    template<class F>
    inline CELER_FUNCTION Intersection visit_leaf(
        BvhNodeId leaf_node_id, Intersection intersection, F&& visit_vol) const;

    // Calculate the current min intersection, which may/may not be in inf_vols
    template<class F>
    inline CELER_FUNCTION Intersection visit_inf_vols(
        Intersection intersection, F&& visit_vol) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector a of bounding boxes and storage.
 */
CELER_FUNCTION
BvhIntersectingVolFinder::BvhIntersectingVolFinder(
    BvhTreeRecord const& tree, BvhIntersectingVolFinder::Storage const& storage)
    : view_(tree, storage)
{
    CELER_EXPECT(tree);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the minimum intersection, with supplied maximum search distance.
 *
 * If no intersection is found within max_search_dist, an empty Intersection
 * object is returned. The visit_vol argument should be of the form:
 *
 * detail::Intersection(*)(LocalVolumeId id, real_type max_search_dist)
 *
 * Other information required by the functor should be handled through
 * lambda capture.
 */
template<class F>
CELER_FUNCTION auto
BvhIntersectingVolFinder::operator()(BvhIntersectingVolFinder::Ray ray,
                                     F&& visit_vol,
                                     real_type max_search_dist) const
    -> Intersection
{
    using Side = BvhInternalNode::Side;

    Intersection intersection{OnLocalSurface{}, max_search_dist};

    // Stack of deferred nodes
    using StackT = IdStack<BvhNodeId, max_bvh_depth - 1>;
    BvhNodeId stack_spill_[StackT::spill_extent];
    StackT stack{stack_spill_};
    static_assert(stack.capacity() == max_bvh_depth);
    stack.push(BvhNodeId{0});

    while (!stack.empty())
    {
        if (!view_.is_internal(stack.top()))
        {
            intersection
                = this->visit_leaf(stack.top(), intersection, visit_vol);
            stack.pop();
            continue;
        }

        auto const& node = view_.inner_node(stack.top());
        stack.pop();
        int ax = to_int(node.axis());

        // Guess the better edge to traverse first: unrolled with loads for
        // GPU performance
        FastBBox first_bbox = node.bbox(Side::left);
        BvhNodeId first_child = node.child(Side::left);
        FastBBox second_bbox = node.bbox(Side::right);
        BvhNodeId second_child = node.child(Side::right);

        if (ray.pos[ax] > node.bbox(Side::right).lower()[ax])
        {
            trivial_swap(first_bbox, second_bbox);
            trivial_swap(first_child, second_child);
        }

        // Choose the next node on the basis of which edges are hits
        if (this->visit_bbox(second_bbox, ray, intersection.distance))
        {
            stack.push(second_child);
        }
        if (this->visit_bbox(first_bbox, ray, intersection.distance))
        {
            stack.push(first_child);
        }
    }

    return this->visit_inf_vols(intersection, visit_vol);
}
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Determine if the intersection with an edge/vol bbox is less than min_dist.
 */
CELER_FUNCTION
bool BvhIntersectingVolFinder::visit_bbox(
    FastBBox const& bbox, Ray ray, real_type min_dist) const
{
    return intersects_segment(bbox, ray.pos, ray.dir, min_dist);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the current min intersection, which may/may not be on this leaf.
 */
template<class F>
CELER_FUNCTION auto BvhIntersectingVolFinder::visit_leaf(
    BvhNodeId leaf_node_id, Intersection min_intersection, F&& visit_vol) const
    -> Intersection
{
    for (auto id : view_.leaf_vol_ids(leaf_node_id))
    {
        auto intersection = visit_vol(id, min_intersection.distance);
        if (intersection)
        {
            CELER_ASSERT(intersection.distance <= min_intersection.distance);
            min_intersection = intersection;
        }
    }
    return min_intersection;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the current min intersection, which may/may not be in inf_vols.
 */
template<class F>
CELER_FUNCTION auto BvhIntersectingVolFinder::visit_inf_vols(
    Intersection min_intersection, F&& visit_vol) const -> Intersection
{
    for (auto id : view_.inf_vol_ids())
    {
        auto intersection = visit_vol(id, min_intersection.distance);
        if (intersection)
        {
            CELER_ASSERT(intersection.distance <= min_intersection.distance);
            min_intersection = intersection;
        }
    }
    return min_intersection;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
