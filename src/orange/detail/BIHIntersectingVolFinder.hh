//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHIntersectingVolFinder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Algorithms.hh"

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
 * Traverse BIH tree using a depth-first search.
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
        Real3 pos;
        Real3 dir;
    };

    //!@}

    // Construct from vector of bounding boxes and storage for LocalVolumeIds
    inline CELER_FUNCTION
    BIHIntersectingVolFinder(BIHTree const& tree, Storage const& storage);

    // Point-in-volume operation
    template<class F>
    inline CELER_FUNCTION Intersection operator()(Ray const& ray,
                                                  F&& visit_vol) const;

  private:
    //// DATA ////
    BIHView view_;

    //// HELPER FUNCTIONS ////

    // Get the ID of the next node in the traversal sequence
    inline CELER_FUNCTION BIHNodeId next_node(BIHNodeId const& current_id,
                                              BIHNodeId const& previous_id,
                                              Ray const& ray,
                                              double min_dist) const;

    // Determine if an edge or volume bbox
    inline CELER_FUNCTION bool
    visit_bbox(FastBBox const& bbox, Ray const& ray, double min_dist) const;

    // Determine if any leaf node volumes contain the point
    template<class F>
    inline CELER_FUNCTION Intersection visit_leaf(BIHLeafNode const& leaf_node,
                                                  Ray const& ray,
                                                  Intersection intersection,
                                                  F&& visit_vol) const;

    // Determine if any inf_vols contain the point
    template<class F>
    inline CELER_FUNCTION Intersection visit_inf_vols(Intersection intersection,
                                                      F&& visit_vol) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage.
 */
CELER_FUNCTION
BIHIntersectingVolFinder::BIHIntersectingVolFinder(
    BIHTree const& tree, BIHIntersectingVolFinder::Storage const& storage)
    : view_(tree, storage)
{
    CELER_EXPECT(tree);
}

//---------------------------------------------------------------------------//
/*!
 * Point-in-volume operation.
 */
template<class F>
CELER_FUNCTION auto
BIHIntersectingVolFinder::operator()(BIHIntersectingVolFinder::Ray const& ray,
                                     F&& visit_vol) const -> Intersection
{
    BIHNodeId previous_node;
    BIHNodeId current_node{0};

    Intersection intersection{OnLocalSurface{},
                              std::numeric_limits<real_type>::infinity()};

    do
    {
        if (!view_.is_inner(current_node))
        {
            intersection = this->visit_leaf(
                view_.leaf_node(current_node), ray, intersection, visit_vol);
        }

        previous_node = exchange(
            current_node,
            this->next_node(
                current_node, previous_node, ray, intersection.distance));

    } while (current_node);

    return this->visit_inf_vols(intersection, visit_vol);
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 *  Get the ID of the next node in the traversal sequence.
 */
CELER_FUNCTION
BIHNodeId BIHIntersectingVolFinder::next_node(BIHNodeId const& current_id,
                                              BIHNodeId const& previous_id,
                                              Ray const& ray,
                                              double min_dist) const
{
    using Side = BIHInnerNode::Side;

    BIHNodeId next_id;

    if (view_.is_inner(current_id))
    {
        auto const& current_node = view_.inner_node(current_id);
        auto const& l_edge = current_node.edges[Side::left];
        auto const& r_edge = current_node.edges[Side::right];

        if (previous_id == current_node.parent)
        {
            // Visiting this inner node for the first time; go down either left
            // or right edge
            next_id = this->visit_bbox(l_edge.bbox, ray, min_dist)
                          ? l_edge.child
                          : r_edge.child;
        }
        else if (previous_id == current_node.edges[Side::left].child)
        {
            // Visiting this inner node for the second time; go down right edge
            // or return to parent
            next_id = this->visit_bbox(r_edge.bbox, ray, min_dist)
                          ? r_edge.child
                          : current_node.parent;
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
bool BIHIntersectingVolFinder::visit_bbox(FastBBox const& bbox,
                                          Ray const& ray,
                                          double min_dist) const
{
    return is_inside(bbox, ray.pos)
           || calc_dist_to_inside(bbox, ray.pos, ray.dir) < min_dist;
}

//---------------------------------------------------------------------------//
/*!
 * Determine if any leaf node volumes contain the point.
 */
template<class F>
CELER_FUNCTION auto
BIHIntersectingVolFinder::visit_leaf(BIHLeafNode const& leaf_node,
                                     BIHIntersectingVolFinder::Ray const& ray,
                                     Intersection min_intersection,
                                     F&& visit_vol) const -> Intersection
{
    for (auto id : view_.leaf_volids(leaf_node))
    {
        auto const& bbox = view_.bbox(id);

        if (this->visit_bbox(bbox, ray, min_intersection.distance))
        {
            auto intersection = visit_vol(id);
            if (intersection.distance < min_intersection.distance)
            {
                min_intersection = intersection;
            }
        }
    }
    return min_intersection;
}

//---------------------------------------------------------------------------//
/*!
 * Determine if any volumes in inf_vols contain the point.
 */
template<class F>
CELER_FUNCTION auto
BIHIntersectingVolFinder::visit_inf_vols(Intersection min_intersection,
                                         F&& visit_vol) const -> Intersection
{
    for (auto id : view_.inf_volids())
    {
        auto intersection = visit_vol(id);
        if (intersection.distance < min_intersection.distance)
        {
            min_intersection = intersection;
        }
    }
    return min_intersection;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
