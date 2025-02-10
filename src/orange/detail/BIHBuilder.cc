//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Construct from a Storage object.
 */
BIHBuilder::BIHBuilder(Storage* storage)
    : bboxes_{&storage->bboxes}
    , local_volume_ids_{&storage->local_volume_ids}
    , inner_nodes_{&storage->inner_nodes}
    , leaf_nodes_{&storage->leaf_nodes}
{
    CELER_EXPECT(storage);
}

//---------------------------------------------------------------------------//
/*!
 * Create BIH Nodes.
 */
BIHTree
BIHBuilder::operator()(VecBBox&& bboxes,
                       BIHBuilder::SetLocalVolId const& implicit_vol_ids)
{
    CELER_EXPECT(!bboxes.empty());

    // Store bounding boxes and their corresponding centers
    temp_.bboxes = std::move(bboxes);
    temp_.centers.resize(temp_.bboxes.size());
    std::transform(temp_.bboxes.begin(),
                   temp_.bboxes.end(),
                   temp_.centers.begin(),
                   &celeritas::calc_center<fast_real_type>);

    // Separate infinite bounding boxes from finite
    VecIndices indices;
    VecIndices inf_vol_ids;
    for (auto i : range(temp_.bboxes.size()))
    {
        LocalVolumeId id(i);

        if (implicit_vol_ids.find(id) != implicit_vol_ids.end())
        {
            // Background volume, do not include bbox in tree
        }
        else if (is_infinite(temp_.bboxes[i]))
        {
            // Infinite in *every* direction
            /*!
             * \todo make an exception for "EXTERIOR" volume and remove the
             * "infinite volume" exceptions?
             */
            inf_vol_ids.push_back(id);
        }
        else
        {
            // Prohibit semi-infinite bounding boxes because those break the
            // cost function
            CELER_ASSERT(is_finite(temp_.bboxes[i]));
            indices.push_back(id);
        }
    }

    BIHTree tree;

    tree.bboxes = ItemMap<LocalVolumeId, FastBBoxId>(
        bboxes_.insert_back(temp_.bboxes.begin(), temp_.bboxes.end()));

    tree.inf_vol_ids = local_volume_ids_.insert_back(inf_vol_ids.begin(),
                                                     inf_vol_ids.end());

    if (!indices.empty())
    {
        VecNodes nodes;
        auto inf_bbox = FastBBox::from_infinite();
        this->construct_tree(indices, &nodes, BIHNodeId{}, inf_bbox);
        auto [inner_nodes, leaf_nodes] = this->arrange_nodes(std::move(nodes));

        tree.inner_nodes
            = inner_nodes_.insert_back(inner_nodes.begin(), inner_nodes.end());

        tree.leaf_nodes
            = leaf_nodes_.insert_back(leaf_nodes.begin(), leaf_nodes.end());
    }
    else
    {
        // Degenerate case where all bounding boxes are infinite. Create a
        // single empty leaf node, so that the existence of leaf nodes does not
        // need to be checked at runtime.
        BIHLeafNode const empty_nodes[] = {{}};
        tree.leaf_nodes = leaf_nodes_.insert_back(std::begin(empty_nodes),
                                                  std::end(empty_nodes));
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
                                BIHNodeId parent,
                                FastBBox const& bbox)
{
    using Side = BIHInnerNode::Side;

    auto current_index = nodes->size();
    nodes->resize(nodes->size() + 1);

    BIHPartitioner partition(&temp_.bboxes, &temp_.centers);

    if (auto p = partition(indices))
    {
        BIHInnerNode node;
        node.parent = parent;
        node.axis = p.axis;

        // Return the current bounding box, clipped by the supplied halfspace
        auto get_shrunk_bbox
            = [&bbox, axis = p.axis](Bound bound, FastBBox::real_type pos) {
                  auto out_box = bbox;
                  out_box.shrink(bound, axis, pos);
                  return out_box;
              };

        // Populate left/right bounding planes
        auto left_pos = p.bboxes[Side::left].upper()[to_int(p.axis)];
        node.edges[Side::left].bounding_plane_pos = left_pos;
        node.edges[Side::left].bbox = get_shrunk_bbox(Bound::hi, left_pos);

        auto right_pos = p.bboxes[Side::right].lower()[to_int(p.axis)];
        node.edges[Side::right].bounding_plane_pos = right_pos;
        node.edges[Side::right].bbox = get_shrunk_bbox(Bound::lo, right_pos);

        // Recursively construct the left and right branches
        for (auto side : range(Side::size_))
        {
            node.edges[side].child = BIHNodeId(nodes->size());
            this->construct_tree(p.indices[side],
                                 nodes,
                                 BIHNodeId(current_index),
                                 node.edges[side].bbox);
        }

        CELER_EXPECT(node);
        (*nodes)[current_index] = node;
    }
    else
    {
        BIHLeafNode node;
        node.parent = parent;
        node.vol_ids
            = local_volume_ids_.insert_back(indices.begin(), indices.end());

        CELER_EXPECT(node);
        (*nodes)[current_index] = node;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Separate inner nodes from leaf nodes and renumber accordingly.
 */
BIHBuilder::ArrangedNodes BIHBuilder::arrange_nodes(VecNodes const& nodes) const
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
        return id_cast<BIHNodeId>(new_indices[old.unchecked_get()]);
    };

    for (auto& inner_node : inner_nodes)
    {
        for (auto& edge : inner_node.edges)
        {
            edge.child = remapped_id(edge.child);
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
