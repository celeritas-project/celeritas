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
 */
struct BIHInnerNode
{
    using real_type = fast_real_type;

    struct BoundingPlane
    {
        Axis axis;
        real_type position;
    };

    enum class Edge
    {
        left,
        right,
        size_
    };

    BIHNodeId parent;

    EnumArray<Edge, BIHNodeId> children;
    EnumArray<Edge, BoundingPlane> bounding_planes;

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
 * References to host storage while constructing a Bounding Interval Hierarchy
 * tree.
 */
struct BIHStorage
{
    template<class T>
    using Storage = Collection<T, Ownership::value, MemSpace::host>;
    using BBoxStorage = Storage<FastBBox>;
    using LVIStorage = Storage<LocalVolumeId>;
    using InnerNodeStorage = Storage<BIHInnerNode>;
    using LeafNodeStorage = Storage<BIHLeafNode>;

    BBoxStorage* bboxes = nullptr;
    LVIStorage* local_volume_ids = nullptr;
    InnerNodeStorage* inner_nodes = nullptr;
    LeafNodeStorage* leaf_nodes = nullptr;

    explicit CELER_FUNCTION operator bool() const
    {
        return bboxes != nullptr && local_volume_ids != nullptr
               && inner_nodes != nullptr && leaf_nodes != nullptr;
    }
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
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
