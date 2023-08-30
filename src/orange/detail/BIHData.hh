//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "orange/BoundingBox.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Data for a single inner node in a Bounding Interval Hierarchy.
 *
 * Note that the LEFT bounding plane position is the far right boundary of the
 * left side of the tree, and the RIGHT bounding plane position is the far left
 * boundary of the right side of the tree. Since the halfspaces of created by
 * the bounding planes may overlap, the LEFT bounding plane position could be
 * either left or right of the RIGHT bounding plane position.
 */
struct BIHInnerNode
{
    using real_type = fast_real_type;

    struct BoundingPlane
    {
        real_type position;
        BIHNodeId child;
    };

    enum class Edge
    {
        left,
        right,
        size_
    };

    BIHNodeId parent;

    Axis axis;
    EnumArray<Edge, BoundingPlane> bounding_planes;

    explicit CELER_FUNCTION operator bool() const
    {
        return this->bounding_planes[Edge::left].child
               && this->bounding_planes[Edge::right].child;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for a single leaf node in a Bounding Interval Hierarchy.
 */
struct BIHLeafNode
{
    BIHNodeId parent;

    ItemRange<LocalVolumeId> vol_ids;

    explicit CELER_FUNCTION operator bool() const { return !vol_ids.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Bounding Interval Hierarchy tree.
 */
struct BIHTree
{
    // All bounding boxes managed by the BIH
    ItemMap<LocalVolumeId, FastBBoxId> bboxes;

    // Inner nodes, the first being the root
    ItemRange<BIHInnerNode> inner_nodes;

    // Leaf nodes
    ItemRange<BIHLeafNode> leaf_nodes;

    // VolumeIds for which bboxes have infinite extents, and are therefore
    // note included in the tree
    ItemRange<LocalVolumeId> inf_volids;

    explicit CELER_FUNCTION operator bool() const
    {
        if (!inner_nodes.empty())
        {
            return !bboxes.empty() && !leaf_nodes.empty();
        }
        else
        {
            // Degenerate single leaf node case. This occurs when a tree
            // contains either:
            // a) a single volume
            // b) muliple non-partitionable volumes,
            // b) only infinite volumes.
            return !bboxes.empty() && leaf_nodes.size() == 1;
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
