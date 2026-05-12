//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/IdStack.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"

#include "BIHView.hh"
#include "../BoundingBoxUtils.hh"
#include "../OrangeData.hh"
#include "../univ/detail/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Traverse the BIH to the find the volume that the ray intersects with first.
 *
 * Traversal is carried out using a depth first search. During traversal, the
 * minimum intersection is stored.  The decision to traverse an edge is done by
 * calculating the distance to intersection with the precomputed edge bounding
 * box. The edge bounding box is the bounding box created by clipping an
 * infinite bounding box with all bounding planes between the root node and the
 * current edge (inclusive). If a ray's intersection with the edge bbox is
 * found to be nearer than the current minimum intersection, traversal proceeds
 * down that edge. Likewise, when a root node is reacted, intersections with
 * volume bboxes are first tested against the minimum intersection prior to
 * testing the the volume itself. The minimum intersection is only modified
 * when a nearer minimumium intersection with a actual volume if found, NOT a
 * nearer intersection with an edge bbox or volume bbox. This is because is is
 * possible to have a ray that intersects with a volume's bbox, but not the
 * volume itself.
 *
 * \todo move to top-level orange directory out of detail namespace
 */
class BIHIntersectingVolFinder
{
  public:
    //!@{
    //! \name Type aliases
    using Storage = NativeCRef<BIHTreeData>;

    struct Ray
    {
        Real3 const& pos;
        Real3 const& dir;
    };
    //!@}

    // Construct from a vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION BIHIntersectingVolFinder(BIHTreeRecord const& tree,
                                                   Storage const& storage);

    // Calculate the minimum intersection, with supplied maximum search
    // distance
    template<class F>
    inline CELER_FUNCTION Intersection
    operator()(Ray ray, F&& visit_vol, real_type max_search_dist) const;

  private:
    //// DATA ////
    BIHView view_;

    //// HELPER FUNCTIONS ////

    // Determine if the intersection with an edge/vol bbox is less than
    // min_dist
    inline CELER_FUNCTION bool
    visit_bbox(FastBBox const& bbox, Ray ray, real_type min_dist) const;

    // Calculate the current min intersection, which may/may not be on this
    // leaf
    template<class F>
    inline CELER_FUNCTION Intersection visit_leaf(BIHLeafNode const& leaf_node,
                                                  Ray ray,
                                                  Intersection intersection,
                                                  F&& visit_vol) const;

    // Calculate the current min intersection, which may/may not be in inf_vols
    template<class F>
    inline CELER_FUNCTION Intersection visit_inf_vols(Intersection intersection,
                                                      F&& visit_vol) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector a of bounding boxes and storage.
 */
CELER_FUNCTION
BIHIntersectingVolFinder::BIHIntersectingVolFinder(
    BIHTreeRecord const& tree, BIHIntersectingVolFinder::Storage const& storage)
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
BIHIntersectingVolFinder::operator()(BIHIntersectingVolFinder::Ray ray,
                                     F&& visit_vol,
                                     real_type max_search_dist) const
    -> Intersection
{
    using Side = BIHInnerNode::Side;

    Intersection intersection{OnLocalSurface{}, max_search_dist};

    // Stack of deferred nodes
    using StackT = IdStack<BIHNodeId, max_bih_depth - 1>;
    BIHNodeId stack_spill_[StackT::spill_extent];
    StackT stack{stack_spill_};
    static_assert(stack.capacity() == max_bih_depth);
    stack.push(BIHNodeId{0});

    while (!stack.empty())
    {
        if (!view_.is_inner(stack.top()))
        {
            intersection = this->visit_leaf(
                view_.leaf_node(stack.top()), ray, intersection, visit_vol);
            stack.pop();
            continue;
        }

        auto const& node = view_.inner_node(stack.top());
        stack.pop();
        int ax = to_int(node.axis);

        // Guess the better edge to traverse first
        auto first_edge = node.edges[Side::left];
        auto second_edge = node.edges[Side::right];

        bool skip_first
            = (ray.dir[ax] >= 0)
              && (ray.pos[ax] > node.edges[Side::left].bounding_plane_pos);
        bool skip_second
            = (ray.dir[ax] <= 0)
              && (ray.pos[ax] < node.edges[Side::right].bounding_plane_pos);

        if (ray.pos[ax] > node.edges[Side::right].bounding_plane_pos)
        {
            trivial_swap(first_edge, second_edge);
            trivial_swap(skip_first, skip_second);
        }

        // Determine if the first and second edges are hits, short circuiting
        // with skip_* before testing bounding boxes
        bool hit_first
            = !skip_first
              && this->visit_bbox(first_edge.bbox, ray, intersection.distance);
        bool hit_second = !skip_second
                          && this->visit_bbox(
                              second_edge.bbox, ray, intersection.distance);

        // Choose the next node on the basis of which edges are hits
        if (hit_second)
        {
            stack.push(second_edge.child);
        }
        if (hit_first)
        {
            stack.push(first_edge.child);
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
bool BIHIntersectingVolFinder::visit_bbox(FastBBox const& bbox,
                                          Ray ray,
                                          real_type min_dist) const
{
    return is_inside(bbox, ray.pos)
           || calc_dist_to_inside(bbox, ray.pos, ray.dir) < min_dist;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the current min intersection, which may/may not be on this leaf.
 */
template<class F>
CELER_FUNCTION auto
BIHIntersectingVolFinder::visit_leaf(BIHLeafNode const& leaf_node,
                                     BIHIntersectingVolFinder::Ray ray,
                                     Intersection min_intersection,
                                     F&& visit_vol) const -> Intersection
{
    for (auto id : view_.leaf_vol_ids(leaf_node))
    {
        auto const& bbox = view_.bbox(id);

        if (this->visit_bbox(bbox, ray, min_intersection.distance))
        {
            auto intersection = visit_vol(id, min_intersection.distance);
            if (intersection)
            {
                CELER_ASSERT(intersection.distance
                             <= min_intersection.distance);
                min_intersection = intersection;
            }
        }
    }
    return min_intersection;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the current min intersection, which may/may not be in inf_vols.
 */
template<class F>
CELER_FUNCTION auto
BIHIntersectingVolFinder::visit_inf_vols(Intersection min_intersection,
                                         F&& visit_vol) const -> Intersection
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
