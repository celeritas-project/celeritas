//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHData.hh
//! \todo move to orange/BihTreeData
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "geocel/BoundingBox.hh"

#include "../OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Data for a single inner node in a Bounding Interval Hierarchy.
 *
 * As a convention, a node's LEFT edge corresponds to the half space that is
 * less than the partition value. In other words, the LEFT bounding plane
 * position is the far right boundary of the left side of the tree, and the
 * RIGHT bounding plane position is the far left boundary of the right side of
 * the tree. Since the halfspaces created by the bounding planes may overlap,
 * the LEFT bounding plane position could be either left or right of the RIGHT
 * bounding plane position.
 */
struct BIHInnerNode
{
    using real_type = fast_real_type;

    struct Edge
    {
        //! The position of the bounding plane along the partition axis
        real_type bounding_plane_pos{};
        //! The child node connected to this edge
        BIHNodeId child;
        //! Bbox created by clipping an inf bbox with the bounding planes
        //! between this edge (inclusive) and the root.
        FastBBox bbox;
    };

    enum class Side
    {
        left,
        right,
        size_
    };

    BIHNodeId parent;  //!< Parent node ID
    Axis axis;  //!< Axis that the partition is performed on
    EnumArray<Side, Edge> edges;  //!< Left/right edges

    explicit CELER_FUNCTION operator bool() const
    {
        return this->edges[Side::left].child && this->edges[Side::right].child;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for a single leaf node in a Bounding Interval Hierarchy.
 */
struct BIHLeafNode
{
    BIHNodeId parent;  //!< Parent node ID
    ItemRange<LocalVolumeId> vol_ids;

    explicit CELER_FUNCTION operator bool() const { return !vol_ids.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Bounding Interval Hierarchy tree.
 *
 * Infinite bounding boxes are not included in the main tree.
 *
 * \todo Rename BihTreeRecord
 */
struct BIHTree
{
    //! All bounding boxes managed by the BIH
    ItemMap<LocalVolumeId, FastBBoxId> bboxes;

    //! Inner nodes, the first being the root
    ItemRange<BIHInnerNode> inner_nodes;

    //! Leaf nodes
    ItemRange<BIHLeafNode> leaf_nodes;

    //! Local volumes that have infinite bounding boxes
    ItemRange<LocalVolumeId> inf_vol_ids;

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
            // b) multiple non-partitionable volumes,
            // b) only infinite volumes.
            return !bboxes.empty() && leaf_nodes.size() == 1;
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
