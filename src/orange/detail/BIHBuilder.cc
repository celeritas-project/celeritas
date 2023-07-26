//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.cc
//---------------------------------------------------------------------------//
#include "BIHBuilder.hh"

#include "orange/BoundingBoxUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Variadic-templated struct for overloaded operator() calls.
 */
template<typename... Ts>
struct Overload : Ts...
{
    using Ts::operator()...;
};

//---------------------------------------------------------------------------//
/*!
 * "Deduction guide" for instantiating Overload objects w/o specifying types.
 */
template<class... Ts>
Overload(Ts&&...) -> Overload<Ts...>;
}  // namespace
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from vector of bounding boxes and storage for LocalVolumeIds.
 */
BIHBuilder::BIHBuilder(VecBBox bboxes,
                       BIHBuilder::BboxStorage* bbox_storage,
                       BIHBuilder::LVIStorage* lvi_storage,
                       BIHBuilder::InnerNodeStorage* inner_node_storage,
                       BIHBuilder::LeafNodeStorage* leaf_node_storage)
    : bboxes_(std::move(bboxes))
    , bbox_storage_(bbox_storage)
    , lvi_storage_(lvi_storage)
    , inner_node_storage_(inner_node_storage)
    , leaf_node_storage_(leaf_node_storage)
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
BIHTree BIHBuilder::operator()() const
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

    auto [inner_nodes, leaf_nodes] = this->arrange_nodes(std::move(nodes));

    BIHTree params;

    params.bboxes = ItemMap<LocalVolumeId, BoundingBoxId>(
        make_builder(bbox_storage_).insert_back(bboxes_.begin(), bboxes_.end()));

    params.inner_nodes
        = make_builder(inner_node_storage_)
              .insert_back(inner_nodes.begin(), inner_nodes.end());
    params.leaf_nodes = make_builder(leaf_node_storage_)
                            .insert_back(leaf_nodes.begin(), leaf_nodes.end());

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
    auto current_index = nodes.size();
    nodes.resize(nodes.size() + 1);

    if (partitioner_.is_partitionable(indices))
    {
        BIHInnerNode node;
        node.parent = parent;

        auto p = partitioner_(indices);
        auto ax = to_int(p.axis);

        BIHInnerNode::BoundingPlane left_plane{
            p.axis,
            static_cast<BIHInnerNode::position_type>(p.left_bbox.upper()[ax])};

        BIHInnerNode::BoundingPlane right_plane{
            p.axis,
            static_cast<BIHInnerNode::position_type>(p.right_bbox.lower()[ax])};

        node.bounding_planes = {left_plane, right_plane};

        // Recursively construct the left and right branches
        node.children[BIHInnerNode::Edge::left] = BIHNodeId(nodes.size());
        this->construct_tree(p.left_indices, nodes, BIHNodeId(current_index));

        node.children[BIHInnerNode::Edge::right] = BIHNodeId(nodes.size());
        this->construct_tree(p.right_indices, nodes, BIHNodeId(current_index));

        CELER_EXPECT(node);
        nodes[current_index] = node;
    }
    else
    {
        BIHLeafNode node;
        node.parent = parent;
        node.vol_ids = make_builder(lvi_storage_)
                           .insert_back(indices.begin(), indices.end());

        CELER_EXPECT(node);
        nodes[current_index] = node;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Seperate nodes into inner and leaf vectors and renumber accordingly.
 */
BIHBuilder::ArrangedNodes BIHBuilder::arrange_nodes(VecNodes nodes) const
{
    VecInnerNodes inner_nodes;
    VecLeafNodes leaf_nodes;

    std::vector<bool> is_leaf;
    std::vector<BIHNodeId> new_ids;

    auto visit_node
        = Overload{[&](BIHInnerNode const& node) {
                       new_ids.push_back(BIHNodeId(inner_nodes.size()));
                       inner_nodes.push_back(node);
                       is_leaf.push_back(false);
                   },
                   [&](BIHLeafNode const& node) {
                       new_ids.push_back(BIHNodeId(leaf_nodes.size()));
                       leaf_nodes.push_back(node);
                       is_leaf.push_back(true);
                   }};

    for (auto const& node : nodes)
    {
        std::visit(visit_node, node);
    }

    size_type offset = inner_nodes.size();

    for (auto i : range(nodes.size()))
    {
        if (is_leaf[i])
        {
            new_ids[i] = new_ids[i] + offset;
        }
    }

    for (auto& inner_node : inner_nodes)
    {
        inner_node.children[BIHInnerNode::Edge::left]
            = new_ids[inner_node.children[BIHInnerNode::Edge::left]
                          .unchecked_get()];
        inner_node.children[BIHInnerNode::Edge::right]
            = new_ids[inner_node.children[BIHInnerNode::Edge::right]
                          .unchecked_get()];

        // Handle root node
        if (inner_node.parent)
        {
            inner_node.parent = new_ids[inner_node.parent.unchecked_get()];
        }
    }

    for (auto& leaf_node : leaf_nodes)
    {
        // Handle where the entire tree is a single leaf node
        if (leaf_node.parent)
        {
            leaf_node.parent = new_ids[leaf_node.parent.unchecked_get()];
        }
    }

    return std::make_pair(inner_nodes, leaf_nodes);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
