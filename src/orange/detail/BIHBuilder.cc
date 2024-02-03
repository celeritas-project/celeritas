//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.cc
//---------------------------------------------------------------------------//
#include "BIHBuilder.hh"

#include "corecel/cont/VariantUtils.hh"

#include "BIHUtils.hh"
#include "../BoundingBoxUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a Storage object
 */
BIHBuilder::BIHBuilder(Storage* storage) : storage_(storage)
{
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Create BIH Nodes.
 */
BIHTree BIHBuilder::operator()(VecBBox bboxes)
{
    CELER_EXPECT(!bboxes.empty());

    // Store bounding boxes and their corresponding centers
    bboxes_ = std::move(bboxes);
    centers_.resize(bboxes_.size());
    std::transform(bboxes_.begin(),
                   bboxes_.end(),
                   centers_.begin(),
                   &celeritas::calc_center<fast_real_type>);

    // Separate infinite bounding boxes from finite
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
            CELER_ASSERT(bboxes_[i]);
            inf_volids.push_back(id);
        }
    }

    BIHTree tree;

    tree.bboxes = ItemMap<LocalVolumeId, FastBBoxId>(
        make_builder(&storage_->bboxes)
            .insert_back(bboxes_.begin(), bboxes_.end()));

    tree.inf_volids = make_builder(&storage_->local_volume_ids)
                          .insert_back(inf_volids.begin(), inf_volids.end());

    if (!indices.empty())
    {
        VecNodes nodes;
        this->construct_tree(indices, &nodes, BIHNodeId{});
        auto [inner_nodes, leaf_nodes] = this->arrange_nodes(std::move(nodes));

        tree.inner_nodes
            = make_builder(&storage_->inner_nodes)
                  .insert_back(inner_nodes.begin(), inner_nodes.end());

        tree.leaf_nodes
            = make_builder(&storage_->leaf_nodes)
                  .insert_back(leaf_nodes.begin(), leaf_nodes.end());
    }
    else
    {
        // Degenerate case where all bounding boxes are infinite. Create a
        // single empty leaf node, so that the existence of leaf nodes does not
        // need to be checked at runtime.
        VecLeafNodes leaf_nodes{1};
        tree.leaf_nodes
            = make_builder(&storage_->leaf_nodes)
                  .insert_back(leaf_nodes.begin(), leaf_nodes.end());
    }

    return tree;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Recursively construct BIH nodes for a vector of bbox indices.
 */
void BIHBuilder::construct_tree(VecIndices const& indices,
                                VecNodes* nodes,
                                BIHNodeId parent) const
{
    using Edge = BIHInnerNode::Edge;

    auto current_index = nodes->size();
    nodes->resize(nodes->size() + 1);

    BIHPartitioner partition(&bboxes_, &centers_);

    if (auto p = partition(indices))
    {
        BIHInnerNode node;
        node.parent = parent;
        node.axis = p.axis;

        auto ax = to_int(p.axis);

        node.bounding_planes[Edge::left].position
            = p.bboxes[Edge::left].upper()[ax];
        node.bounding_planes[Edge::right].position
            = p.bboxes[Edge::right].lower()[ax];

        // Recursively construct the left and right branches
        for (auto edge : range(Edge::size_))
        {
            node.bounding_planes[edge].child = BIHNodeId(nodes->size());
            this->construct_tree(
                p.indices[edge], nodes, BIHNodeId(current_index));
        }

        CELER_EXPECT(node);
        (*nodes)[current_index] = node;
    }
    else
    {
        BIHLeafNode node;
        node.parent = parent;
        node.vol_ids = make_builder(&storage_->local_volume_ids)
                           .insert_back(indices.begin(), indices.end());

        CELER_EXPECT(node);
        (*nodes)[current_index] = node;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Separate inner nodes from leaf nodes and renumber accordingly.
 */
BIHBuilder::ArrangedNodes BIHBuilder::arrange_nodes(VecNodes nodes) const
{
    VecInnerNodes inner_nodes;
    VecLeafNodes leaf_nodes;

    std::vector<bool> is_leaf;
    std::vector<std::size_t> new_indices;

    is_leaf.reserve(nodes.size());
    new_indices.reserve(nodes.size());

    auto insert_node = Overload{[&](BIHInnerNode const& node) {
                                    new_indices.push_back(inner_nodes.size());
                                    inner_nodes.push_back(node);
                                    is_leaf.push_back(false);
                                },
                                [&](BIHLeafNode const& node) {
                                    new_indices.push_back(leaf_nodes.size());
                                    leaf_nodes.push_back(node);
                                    is_leaf.push_back(true);
                                }};
    for (auto const& node : nodes)
    {
        std::visit(insert_node, node);
    }

    // Transform "leaf ID" to "node ID"
    auto offset = inner_nodes.size();
    for (auto i : range(nodes.size()))
    {
        if (is_leaf[i])
        {
            new_indices[i] += offset;
        }
    }

    // Remap IDs. "parent" will only be undefined for the root node.
    auto remapped_id = [&new_indices](BIHNodeId old) {
        CELER_EXPECT(old < new_indices.size());
        return BIHNodeId{
            static_cast<size_type>(new_indices[old.unchecked_get()])};
    };

    for (auto& inner_node : inner_nodes)
    {
        for (auto& bp : inner_node.bounding_planes)
        {
            bp.child = remapped_id(bp.child);
        }
        if (inner_node.parent)
        {
            inner_node.parent = remapped_id(inner_node.parent);
        }
    }

    for (auto& leaf_node : leaf_nodes)
    {
        if (leaf_node.parent)
        {
            leaf_node.parent = remapped_id(leaf_node.parent);
        }
    }

    return {std::move(inner_nodes), std::move(leaf_nodes)};
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
