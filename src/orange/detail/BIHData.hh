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
 */
struct BIHInnerNode
{
    using location_type = float;

    struct BoundingPlane
    {
        Axis axis;
        location_type location;
    };

    enum Edge : size_type
    {
        left = 0,
        right = 1
    };

    BIHNodeId parent;

    // inner node only
    Array<BIHNodeId, 2> children;
    Array<BoundingPlane, 2> bounding_planes;

    explicit CELER_FUNCTION operator bool() const
    {
        return this->children[Edge::left] && this->children[Edge::right];
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

    //! True if either a valid inner or leaf node
    explicit CELER_FUNCTION operator bool() const { return !vol_ids.empty(); }
};

//---------------------------------------------------------------------------//
/*!
 * Data for a Bounding Interval Hierarchy.
 */
struct BIHParams
{
    // All bounding boxes managed by the BIH
    ItemMap<LocalVolumeId, BoundingBoxId> bboxes;

    // Inner nodes, the first being the root
    ItemRange<BIHInnerNode> inner_nodes;

    // Leaf nodes
    ItemRange<BIHLeafNode> leaf_nodes;

    // VolumeIds for which bboxes have infinite extents, and are therefore
    // note included in the tree
    ItemRange<LocalVolumeId> inf_volids;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
