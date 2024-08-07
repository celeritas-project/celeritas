//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/DeMorganSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Implement DeMorgan transformations on a \c CsgTree.
 *
 * This class serves as an helper for \c
 * celeritas::orangeinp::transform_negated_joins implementation. It applied
 * DeMorgan's law on a \c CsgTree so that all negations of a set operator are
 * removed and transformed into their equivalent operation.
 *
 * The purpose of this class is to hold the data structures necessary for the
 * simplification and validate operations applied on them. The steering is done
 * by the user.
 *
 * Instances of this class shouldn't outlive the \c CsgTree used to construct
 * it as we keep a reference to it. Do not modify the original tree during the
 * simplification.
 */
class DeMorganSimplifier
{
  public:
    //!@{
    //! \name Set of NodeId
    using NodeIdSet = std::unordered_set<NodeId>;
    //! \name Map of NodeId, caching a pointer to the concrete type of a Node
    //! in the tree
    template<class T>
    using CachedNodeMap = std::unordered_map<NodeId, T const*>;
    //!@}

    //! Construct a simplifier for the given tree
    explicit DeMorganSimplifier(CsgTree const& tree) : tree_(tree) {}

    // Unmark and node and its descendents
    inline void remove_orphaned(NodeId);

    // Signal that this node is a negated operation and that it should be
    // simplified
    inline void mark_negated_operator(NodeId);

    // Handle a negated Joined node
    inline CachedNodeMap<Joined>::mapped_type process_negated_node(NodeId);

    // Handle an orphaned Joined node
    inline NodeIdSet::value_type process_orphaned_join_node(NodeId);

    // Handle a Negated node
    inline NodeIdSet::value_type process_orphaned_negate_node(NodeId);

    // Check if a new Negated node must be inserted
    inline NodeIdSet::value_type process_new_negated_node(NodeId);

    // Handle a Negated node with a Joined child
    inline CachedNodeMap<Negated>::mapped_type
        process_transformed_negate_node(NodeId);

  private:
    // Add negated nodes for operands of a Joined node
    inline void add_new_negated_nodes(NodeId);

    //! the tree to simplify
    CsgTree const& tree_;

    //! For each node_id in that set, we'll create a Negated{node_id} node in
    //! the new tree.
    NodeIdSet new_negated_nodes_;

    //! We need to insert a negated version of these Joined nodes.
    //! The value is the Node referred to by the key as to avoid an extra
    //! get_if.
    CachedNodeMap<Joined> negated_join_nodes_;

    //! Contains the node_id of Negated{Joined}. These don't need to be
    //! inserted in the new tree and its parent can directly point to the
    //! transformed Joined node.
    CachedNodeMap<Negated> transformed_negated_nodes_;

    //! A Negated{Joined{}} nodes needs to be mapped to the opposite Joined{}
    //! node. Parents need to point to the new Joined{} node. The old Joined{}
    //! node needs to be kept if it had parents other than the Negated node.
    NodeIdSet orphaned_join_nodes_;

    //! Similar to orphaned_join_nodes_, kept in a separate collection are
    //! these require a different handling.
    NodeIdSet orphaned_negate_nodes_;
};

//---------------------------------------------------------------------------//
/*!
 * Nodes passed to \c mark_negated_operator (i.e. negated set operations) are
 * marked as orphan by default. That is, they should not be included in the
 * simplified tree. However, it is possible to later discover that this node
 * had another parent that is not a \c Negated node, therefore we should
 * include that node and its descendents in the simplified tree. This unmark a
 * node previously marked for deletion.
 */
inline void DeMorganSimplifier::remove_orphaned(NodeId node_id)
{
    // we need to recursively unmark orphaned nodes and unmark ourself
    if (auto* node_ptr = &tree_[node_id];
        auto* join_node = std::get_if<orangeinp::Joined>(node_ptr))
    {
        for (auto join_operand : join_node->nodes)
        {
            remove_orphaned(join_operand);
        }
        if (auto item = orphaned_join_nodes_.find(node_id);
            item != orphaned_join_nodes_.end())
        {
            orphaned_join_nodes_.erase(item);
        }
    }
    else if (auto* negate_node = std::get_if<orangeinp::Negated>(node_ptr))
    {
        // Negated{Joined{}} should be simplified, so don't unmark them
        if (!std::get_if<orangeinp::Joined>(&tree_[negate_node->node]))
        {
            remove_orphaned(negate_node->node);
        }
        if (auto item = orphaned_negate_nodes_.find(node_id);
            item != orphaned_negate_nodes_.end())
        {
            orphaned_negate_nodes_.erase(item);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Record a Negated{Joined{}} node that must be transformed to the opposite set
 * operation. Per DeMorgan's law, this will also mark all operands of the set
 * operation as needing needing a Negated{} parent.
 *
 * \param negated_id of a Negated node pointing to Joined node inside tree.
 */
inline void DeMorganSimplifier::mark_negated_operator(NodeId negated_id)
{
    CELER_EXPECT(negated_id < tree_.size());
    auto* negated_node = std::get_if<orangeinp::Negated>(&tree_[negated_id]);
    transformed_negated_nodes_[negated_id] = negated_node;
    CELER_ASSERT(negated_node);
    auto* joined = std::get_if<orangeinp::Joined>(&tree_[negated_node->node]);
    CELER_ASSERT(joined);
    negated_join_nodes_[negated_node->node] = joined;
    add_new_negated_nodes(negated_node->node);
}

//---------------------------------------------------------------------------//
/*!
 * Handle a negated Joined node during simplified tree construction.
 */
inline auto DeMorganSimplifier::process_negated_node(NodeId node_id)
    -> CachedNodeMap<Joined>::mapped_type
{
    if (auto iter = negated_join_nodes_.find(node_id);
        iter != negated_join_nodes_.end())
    {
        auto result = iter->second;
        negated_join_nodes_.erase(iter);
        return result;
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Handle an orphaned \c Joined node during simplified tree construction.
 */
inline auto DeMorganSimplifier::process_orphaned_join_node(NodeId node_id)
    -> NodeIdSet::value_type
{
    if (auto iter = orphaned_join_nodes_.find(node_id);
        iter != orphaned_join_nodes_.end())
    {
        auto result = *iter;
        orphaned_join_nodes_.erase(iter);
        return result;
    }
    return NodeId{};
}

//---------------------------------------------------------------------------//
/*!
 * Handle a \c Negated node during simplified tree construction.
 */
inline auto DeMorganSimplifier::process_orphaned_negate_node(NodeId node_id)
    -> NodeIdSet::value_type
{
    if (auto iter = orphaned_negate_nodes_.find(node_id);
        iter != orphaned_negate_nodes_.end())
    {
        auto result = *iter;
        orphaned_negate_nodes_.erase(iter);
        return result;
    }
    return NodeId{};
}

//---------------------------------------------------------------------------//
/*!
 * Check if a \c Negated node pointing to node_id must be inserted in the
 * simplified tree.
 */
inline auto DeMorganSimplifier::process_new_negated_node(NodeId node_id)
    -> NodeIdSet::value_type
{
    if (auto iter = new_negated_nodes_.find(node_id);
        iter != new_negated_nodes_.end())
    {
        auto result = *iter;
        new_negated_nodes_.erase(iter);
        return result;
    }
    return NodeId{};
}

//---------------------------------------------------------------------------//
/*!
 * Handle a \c Negated node with a \c Joined child during simplified tree
 * construction.
 */
inline auto DeMorganSimplifier::process_transformed_negate_node(NodeId node_id)
    -> CachedNodeMap<Negated>::mapped_type
{
    if (auto iter = transformed_negated_nodes_.find(node_id);
        iter != transformed_negated_nodes_.end())
    {
        auto result = iter->second;
        transformed_negated_nodes_.erase(iter);
        return result;
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Recusively record that we need to insert \c Negated node for operands of a
 * \c Joined node. This handles cas
 *
 * \param node_id a \c Joined node_id with a \c Negated parent
 */
inline void DeMorganSimplifier::add_new_negated_nodes(NodeId node_id)
{
    CELER_EXPECT(std::get_if<orangeinp::Joined>(&tree_[node_id]));
    if (auto* join_node = std::get_if<orangeinp::Joined>(&tree_[node_id]))
    {
        for (auto const& join_operand : join_node->nodes)
        {
            if (auto node_ptr = &tree_[join_operand];
                auto* joined_join_operand
                = std::get_if<orangeinp::Joined>(node_ptr))
            {
                // the new Negated node will point to a Joined node
                // so we need to transform that Joined node as well
                add_new_negated_nodes(join_operand);
                negated_join_nodes_[join_operand] = joined_join_operand;
            }
            else if (auto* negated = std::get_if<orangeinp::Negated>(node_ptr))
            {
                // double negation, this simplifies unless it has
                // another parent
                orphaned_negate_nodes_.insert(join_operand);
                // however, we need to make sure that we're keeping
                // the target of the double negation around because
                // someone might need it
                remove_orphaned(negated->node);
            }

            // per DeMorgran's law, negate each operand of the Joined
            // node if we're not inserting a Negated{Joined{}}.
            if (!std::get_if<orangeinp::Joined>(&tree_[join_operand]))
            {
                new_negated_nodes_.insert(join_operand);
            }
        }
        // assume that the Joined node doesn't have other parents and
        // mark it for deletion. This is done after recusive calls as
        // parents can only have a higher node_id and the recursive
        // calls explore childrens with lower node_id. This can be
        // unmarked later as we process potential parents
        orphaned_join_nodes_.insert(node_id);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas