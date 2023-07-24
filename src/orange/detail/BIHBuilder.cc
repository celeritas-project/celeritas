//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.cc
//---------------------------------------------------------------------------//
#include "BIHBuilder.hh"

#include "BoundingBoxUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage for LocalVolumeIds.
 */
BIHBuilder::BIHBuilder(VecBBox bboxes,
                       BIHBuilder::LVIStorage* lvi_storage,
                       BIHBuilder::NodeStorage* node_storage)
    : bboxes_(bboxes), lvi_storage_(lvi_storage), node_storage_(node_storage)
{
    CELER_EXPECT(!bboxes_.empty());

    centers_.resize(bboxes_.size());
    std::transform(bboxes_.begin(),
                   bboxes_.end(),
                   centers_.begin(),
                   [&](BoundingBox const& bbox) { return center(bbox); });

    partitioner_ = BIHPartitioner(&bboxes_, &centers_);
}

//---------------------------------------------------------------------------//
/*!
 * Create BIH Nodes.
 */
BIHParams BIHBuilder::operator()() const
{
    VecIndices indices;
    VecIndices inf_volids;

    for (auto i : range(bboxes_.size()))
    {
        LocalVolumeId id(i);

        if (!is_infinite(bboxes_[i]))
        {
            indices.push_back(id);
        }
        else
        {
            inf_volids.push_back(id);
        }
    }

    VecNodes nodes;
    this->construct_tree(indices, nodes, static_cast<BIHNodeId>(-1));

    BIHParams params;
    params.nodes
        = make_builder(node_storage_).insert_back(nodes.begin(), nodes.end());
    params.inf_volids = make_builder(lvi_storage_)
                            .insert_back(inf_volids.begin(), inf_volids.end());

    return params;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Recursively construct BIH nodes for a vector of bbox indices.
 */
void BIHBuilder::construct_tree(VecIndices const& indices,
                                VecNodes& nodes,
                                BIHNodeId parent) const
{
    nodes.push_back(BIHNode());
    auto current_index = nodes.size() - 1;
    nodes[current_index].parent = parent;

    if (partitioner_.is_partitionable(indices))
    {
        auto p = partitioner_(indices);
        auto ax = to_int(p.axis);

        BIHNode::BoundingPlane left_plane{
            p.axis,
            static_cast<BIHNode::location_type>(p.left_bbox.upper()[ax])};

        BIHNode::BoundingPlane right_plane{
            p.axis,
            static_cast<BIHNode::location_type>(p.right_bbox.lower()[ax])};

        nodes[current_index].bounding_planes = {left_plane, right_plane};

        // Recursively construct the left and right branches
        nodes[current_index].children[BIHNode::Edge::left]
            = BIHNodeId(nodes.size());
        this->construct_tree(p.left_indices, nodes, BIHNodeId(current_index));

        nodes[current_index].children[BIHNode::Edge::right]
            = BIHNodeId(nodes.size());
        this->construct_tree(p.right_indices, nodes, BIHNodeId(current_index));
    }
    else
    {
        this->make_leaf(nodes[current_index], indices);
    }

    CELER_EXPECT(nodes[current_index]);
}

//---------------------------------------------------------------------------//
/*!
 * Add leaf volume ids to a given node.
 */
void BIHBuilder::make_leaf(BIHNode& node, VecIndices const& indices) const
{
    CELER_EXPECT(!node);
    CELER_EXPECT(!indices.empty());

    node.vol_ids
        = make_builder(lvi_storage_).insert_back(indices.begin(), indices.end());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
