//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.cc
//---------------------------------------------------------------------------//
#include "BIHBuilder.hh"

#include "orange/BoundingBoxUtils.hh"
#include "orange/detail/BIHUtils.hh"

namespace celeritas
{
namespace detail
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
    // Store bounding boxes and their corresponding centers
    CELER_EXPECT(!bboxes.empty());
    bboxes_ = std::move(bboxes);
    centers_.resize(bboxes_.size());
    std::transform(bboxes_.begin(),
                   bboxes_.end(),
                   centers_.begin(),
                   &celeritas::center<fast_real_type>);

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
        // single empty leaf node, so that the existance of leaf nodes does not
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
        for (auto edge : range(BIHInnerNode::Edge::size_))
        {
            inner_node.bounding_planes[edge].child
                = new_ids[inner_node.bounding_planes[edge].child.unchecked_get()];
        }

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

    return {std::move(inner_nodes), std::move(leaf_nodes)};
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
