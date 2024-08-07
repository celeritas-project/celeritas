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
 * Instances of this class shouldn't outlive the \c CsgTree used to construct
 * it as we keep a reference to it. Do not modify the original tree during the
 * simplification.
 */
class DeMorganSimplifier
{
  public:
    //! Construct a simplifier for the given tree
    explicit DeMorganSimplifier(CsgTree const& tree) : tree_(tree) {}

    // Simplify
    inline CsgTree operator()();

  private:
    //!@{
    //! \name Set of NodeId
    using NodeIdSet = std::unordered_set<NodeId>;
    //! \name Map of NodeId, caching a pointer to the concrete type of a Node
    //! in the tree
    template<class T>
    using CachedNodeMap = std::unordered_map<NodeId, T const*>;
    //!@}

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

    // Add negated nodes for operands of a Joined node
    inline void add_new_negated_nodes(NodeId);

    // First pass through the tree to find negated set operations
    inline void find_negated_joins();

    // Second pass through the tree to build the simplified tree
    inline CsgTree build_simplified_tree();

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

    //! Used during construction of the simplified tree to map replaced nodes
    //! in the original tree to their new id in the simplified tree
    std::unordered_map<NodeId, NodeId> inserted_nodes_;

    //! Used during construction of the simplified tree to map unmodified nodes
    //! to their new id in the simplified tree
    std::unordered_map<NodeId, NodeId> original_new_nodes_;
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
 * Simplify
 */
inline CsgTree DeMorganSimplifier::operator()()
{
    // TODO: pass the tree here instead of ctor and clear the state for
    // multi-use
    find_negated_joins();
    return build_simplified_tree();
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
/*!
 * First pass through the tree to find negated set operations
 */
inline void DeMorganSimplifier::find_negated_joins()
{
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        // TODO: visitor
        if (auto* node_ptr = &tree_[node_id];
            auto* negated = std::get_if<orangeinp::Negated>(node_ptr))
        {
            // we don't need to check if this is a potential parent of a
            // Joined{} node marked as orphan as this node will also get
            // simplified.

            if (std::get_if<orangeinp::Joined>(&tree_[negated->node]))
            {
                // This is a Negated{Joined{...}}
                mark_negated_operator(node_id);
            }
        }
        else if (auto* aliased = std::get_if<orangeinp::Aliased>(node_ptr))
        {
            remove_orphaned(aliased->node);  // TODO: handle alias
            // pointing to an alias
            // (pointing to an
            // alias)...
        }
        else if (auto* joined = std::get_if<orangeinp::Joined>(node_ptr))
        {
            for (auto const& join_operand : joined->nodes)
            {
                remove_orphaned(join_operand);
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Second pass through the tree to build the simplified tree.
 *
 * \return the simplified tree
 */
inline CsgTree DeMorganSimplifier::build_simplified_tree()
{
    CsgTree result{};

    // Utility to check the two maps for the new id of a node.
    // maybe we can coalesce them in a single map<NodeId, vector<NodeId>>
    auto replace_node_id = [&](NodeId n) {
        if (auto new_id = inserted_nodes_.find(n);
            new_id != inserted_nodes_.end())
        {
            return new_id->second;
        }
        if (auto new_id = original_new_nodes_.find(n);
            new_id != original_new_nodes_.end())
        {
            return new_id->second;
        }
        return n;
    };

    // We can now build the new tree.
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        // The current node is a Joined{} and we need to insert a Negated
        // version of it.
        if (auto negated_node = process_negated_node(node_id))
        {
            // Insert the opposite join instead, updating the children ids.
            Joined const& join = *negated_node;
            OperatorToken opposite_op = (join.op == logic::land)
                                            ? logic::lor
                                            : logic::OperatorToken::land;
            // Lookup the new id of each operand
            std::vector<NodeId> operands;
            operands.resize(join.nodes.size());
            std::transform(join.nodes.cbegin(),
                           join.nodes.cend(),
                           operands.begin(),
                           replace_node_id);

            auto [new_id, inserted]
                = result.insert(Joined{opposite_op, std::move(operands)});
            inserted_nodes_[node_id] = new_id;
        }

        // this Negated{Join} node doesn't have any other parents, so we don't
        // need to insert its original version
        if (process_orphaned_join_node(node_id))
        {
            continue;
        }
        // This Negated{} node is orphaned, most likely because of
        // a double negation so we don't need to insert it.
        // It still needs to be added in inserted_nodes so that when adding
        // a Joined node, we correctly redirect the operand looking for it to
        // the children
        if (auto orphan_node = process_orphaned_negate_node(node_id))
        {
            inserted_nodes_[node_id] = replace_node_id(std::move(orphan_node));
            continue;
        }

        // this node is a Negated{Join} node, we have already inserted the
        // opposite Joined node
        if (auto negated_node = process_transformed_negate_node(node_id))
        {
            // we don't need to insert the original Negated{Join} node
            // redirect parents looking for this node to the new Joined node.
            inserted_nodes_[node_id]
                = inserted_nodes_.find(negated_node->node)->second;
            continue;
        }

        // this node isn't a transformed Join or Negated node, so we can insert
        // it.
        Node new_node = tree_[node_id];

        // We need to update the childrens' ids in the new tree.
        // TODO: visitor
        if (auto* node_ptr = &new_node;
            auto* negated = std::get_if<orangeinp::Negated>(node_ptr))
        {
            negated->node = replace_node_id(negated->node);
        }
        else if (auto* aliased = std::get_if<orangeinp::Aliased>(node_ptr))
        {
            aliased->node = replace_node_id(aliased->node);
        }
        else if (auto* joined = std::get_if<orangeinp::Joined>(node_ptr))
        {
            for (auto& op : joined->nodes)
            {
                // if the node is a Negated{Joined}, we need to match to a
                // newly inserted node
                if (auto* negated = std::get_if<orangeinp::Negated>(&tree_[op]);
                    negated
                    && std::get_if<orangeinp::Joined>(&tree_[negated->node]))
                {
                    op = replace_node_id(op);
                }
                // otherwise, only search unmodified nodes
                else if (auto new_id = original_new_nodes_.find(op);
                         new_id != original_new_nodes_.end())
                {
                    op = new_id->second;
                }
            }
        }
        auto [new_id, inserted] = result.insert(std::move(new_node));
        // this is recorded in a different map as a node in the original tree
        // can be inserted multiple times in the new tree e.g. a
        // Negated{Joined}, if that Joined node has another parent.
        original_new_nodes_[node_id] = new_id;

        // We might have to insert a negated version of that node
        if (process_new_negated_node(node_id))
        {
            auto [new_negated_node_id, negated_inserted]
                = result.insert(Negated{new_id});
            inserted_nodes_[node_id] = new_negated_node_id;
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas