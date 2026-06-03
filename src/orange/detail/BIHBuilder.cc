//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BIHBuilder.cc
//---------------------------------------------------------------------------//
#include "BIHBuilder.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/Collection.hh"
#include "orange/detail/BIHData.hh"

#include "BIHPartitioner.hh"
#include "../BoundingBoxUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 *
 * \param[in] storage  Struct containing collections of persistent data for
 *                     all BIH trees
 * \param[in] inp      Input options that govern BIH construction, i.e.,
 *                     the maximum leaf size and the recursion depth limit
 */
BIHBuilder::BIHBuilder(Storage* storage, Input inp)
    : bboxes_{&storage->bboxes}
    , local_volume_ids_{&storage->local_volume_ids}
    , internal_nodes_{&storage->internal_nodes}
    , leaf_nodes_{&storage->leaf_nodes}
    , inp_{inp}
{
    CELER_EXPECT(storage);
    CELER_EXPECT(inp_);
    CELER_VALIDATE(inp_.depth_limit > 0 && inp_.depth_limit <= max_bih_depth,
                   << "invalid BIH input depth limit " << inp_.depth_limit
                   << ": must be positive and no more than compile-time "
                      "maximum "
                   << max_bih_depth);
    CELER_VALIDATE(inp_.max_leaf_size > 0,
                   << "invalid BIH max leaf size " << inp_.max_leaf_size << ": "
                   << "must be positive");
    CELER_VALIDATE(inp_.num_part_cands > 0,
                   << "invalid BIH partition candidate count "
                   << inp_.num_part_cands << ": "
                   << "must be positive");
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build a BIH tree for the supplied bounding boxes.
 *
 * \param[in] bboxes            All bounding boxes to be included in the tree
 * \param[in] implicit_vol_ids  The ids of the "background" volumes, to be
 *                              excluded from the tree
 *
 * \return The record of the resultant BIH tree
 */
BIHTreeRecord
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

    BIHTreeRecord tree;

    tree.bboxes = ItemMap<LocalVolumeId, FastBBoxId>(
        bboxes_.insert_back(temp_.bboxes.begin(), temp_.bboxes.end()));

    tree.inf_vol_ids = local_volume_ids_.insert_back(inf_vol_ids.begin(),
                                                     inf_vol_ids.end());

    // The depth of the most embedded node (where 1 is the root node), to be
    // calculated during the recursive construction process
    size_type depth = 0;

    if (!indices.empty())
    {
        // Construct the tree recursively
        VecNodes nodes;
        this->construct_tree(indices, &nodes, 0, depth);
        auto [internal_nodes, leaf_nodes]
            = this->arrange_nodes(std::move(nodes));

        tree.internal_nodes = internal_nodes_.insert_back(
            internal_nodes.begin(), internal_nodes.end());

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

    // Assign metadata for diagnostic purposes
    BIHTreeRecord::Metadata md;
    md.num_finite_bboxes = indices.size();
    md.num_infinite_bboxes = inf_vol_ids.size();
    md.depth = depth;
    tree.metadata = md;

    return tree;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Recursively construct BIH nodes for a vector of bbox indices.
 *
 * \param[in] indices        The indices of the bboxes that will be partitioned
 *                           or placed on a leaf node in this function call
 * \param[in, out] nodes     All nodes constructed so far, to be added to
 * \param[in] current_depth  The recursion depth of this function call
 * \param[in] depth          The maximum recursion depth encountered during the
 *                           full construction process
 */
void BIHBuilder::construct_tree(VecIndices const& indices,
                                VecNodes* nodes,
                                size_type current_depth,
                                size_type& depth)
{
    CELER_EXPECT(current_depth < inp_.depth_limit);

    using Side = BIHInternalNode::Side;

    ++current_depth;
    auto current_index = nodes->size();
    nodes->resize(nodes->size() + 1);

    // Create a single leaf containing all bboxes. This lambda is used only
    // once per call to construct_tree.
    auto make_leaf = [&]() {
        BIHLeafNode node;
        node.vol_ids
            = local_volume_ids_.insert_back(indices.begin(), indices.end());
        CELER_EXPECT(node);
        (*nodes)[current_index] = node;
        depth = std::max(depth, current_depth);
    };

    if (indices.size() <= inp_.max_leaf_size
        || current_depth == inp_.depth_limit)
    {
        // All bboxes fit on a single leaf, or we have reached the depth limit;
        // make a leaf and exit early
        make_leaf();
        return;
    }

    BIHPartitioner partition(temp_.bboxes, temp_.centers, inp_.num_part_cands);
    if (auto p = partition(indices))
    {
        // Create internal node
        BIHInternalNode node;
        node.axis = p.axis;

        // Recursively construct the left and right branches
        for (auto side : range(Side::size_))
        {
            node.edges[side].bbox = p.bboxes[side];

            node.edges[side].child = id_cast<BIHNodeId>(nodes->size());
            this->construct_tree(p.indices[side], nodes, current_depth, depth);
        }

        CELER_ASSERT(node);
        (*nodes)[current_index] = node;
    }
    else
    {
        // Bboxes cannot be partitioned; put them all on a single leaf
        make_leaf();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Separate internal nodes from leaf nodes and renumber accordingly.
 *
 * \param[in] nodes  The interspersed inner and leaf nodes
 *
 * \returns  The separated inner and leaf nodes
 */
BIHBuilder::ArrangedNodes BIHBuilder::arrange_nodes(VecNodes const& nodes) const
{
    VecInnerNodes internal_nodes;
    VecLeafNodes leaf_nodes;

    std::vector<bool> is_leaf;
    std::vector<std::size_t> new_indices;

    is_leaf.reserve(nodes.size());
    new_indices.reserve(nodes.size());

    auto insert_node
        = Overload{[&](BIHInternalNode const& node) {
                       new_indices.push_back(internal_nodes.size());
                       internal_nodes.push_back(node);
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
    auto offset = internal_nodes.size();
    for (auto i : range(nodes.size()))
    {
        if (is_leaf[i])
        {
            new_indices[i] += offset;
        }
    }

    // Remap IDs
    auto remapped_id = [&new_indices](BIHNodeId old) {
        CELER_EXPECT(old < new_indices.size());
        return id_cast<BIHNodeId>(new_indices[old.unchecked_get()]);
    };

    for (auto& inner_node : internal_nodes)
    {
        for (auto& edge : inner_node.edges)
        {
            edge.child = remapped_id(edge.child);
        }
    }

    return {std::move(internal_nodes), std::move(leaf_nodes)};
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
