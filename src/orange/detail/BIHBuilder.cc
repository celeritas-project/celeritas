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
    // Create a vector of indices, excluding index 0 (i.e., the exterior)
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

    if (indices.size() > 1)
    {
        auto p = partitioner_(indices);

        if (p)
        {
            auto ax = to_int(p.axis);

            auto [left_indices, right_indices] = apply_partition(indices, p);

            CELER_EXPECT(!left_indices.empty() && !right_indices.empty());

            BIHNode::BoundingPlane left_plane{
                p.axis,
                static_cast<BIHNode::location_type>(
                    bbox_union(bboxes_, left_indices).upper()[ax])};

            BIHNode::BoundingPlane right_plane{
                p.axis,
                static_cast<BIHNode::location_type>(
                    bbox_union(bboxes_, right_indices).lower()[ax])};

            nodes[current_index].bounding_planes = {left_plane, right_plane};

            // Recursively construct the left and right branches
            nodes[current_index].children[BIHNode::Edge::left]
                = BIHNodeId(nodes.size());
            this->construct_tree(left_indices, nodes, BIHNodeId(current_index));
            nodes[current_index].children[BIHNode::Edge::right]
                = BIHNodeId(nodes.size());
            this->construct_tree(
                right_indices, nodes, BIHNodeId(current_index));
        }
        else
        {
            this->make_leaf(nodes[current_index], indices);
        }
    }
    else
    {
        this->make_leaf(nodes[current_index], indices);
    }

    CELER_EXPECT(nodes[current_index]);
}

//---------------------------------------------------------------------------//
/*!
 * Divide bboxes into left and right branches based on a partition.
 */
BIHBuilder::PairVecIndices
BIHBuilder::apply_partition(VecIndices const& indices,
                            BIHPartitioner::Partition const& p) const
{
    CELER_EXPECT(!indices.empty());

    VecIndices left;
    VecIndices right;

    for (auto i : range(indices.size()))
    {
        if (centers_[indices[i].unchecked_get()][to_int(p.axis)] < p.location)
        {
            left.push_back(indices[i]);
        }
        else
        {
            right.push_back(indices[i]);
        }
    }

    return std::make_pair(left, right);
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
