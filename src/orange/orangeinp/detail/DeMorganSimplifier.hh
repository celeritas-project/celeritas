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

    // Add negated nodes for operands of a Joined node
    inline void add_new_negated_nodes(NodeId);

    // Special handling for a Joined or Negated node
    inline bool process_negated_joined_nodes(NodeId, CsgTree&);

    // First pass through the tree to find negated set operations
    inline void find_negated_joins();

    // Second pass through the tree to build the simplified tree
    inline CsgTree build_simplified_tree();

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

    // Utility to check the two maps for the new id of a node.
    inline NodeId replace_node_id(NodeId);

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
 * Nodes passed to \c mark_negated_operator (i.e. negated set operations) are
 * marked as orphan by default. That is, they should not be included in the
 * simplified tree. However, it is possible to later discover that this node
 * had another parent that is not a \c Negated node, therefore we should
 * include that node and its descendents in the simplified tree. This unmark a
 * node previously marked for deletion.
 */
inline void DeMorganSimplifier::remove_orphaned(NodeId node_id)
{
    std::visit(
        Overload{[&](Joined const& joined) {
                     // we need to recursively unmark orphaned nodes and unmark
                     // ourself
                     for (auto join_operand : joined.nodes)
                     {
                         remove_orphaned(join_operand);
                     }
                     if (auto item = orphaned_join_nodes_.find(node_id);
                         item != orphaned_join_nodes_.end())
                     {
                         orphaned_join_nodes_.erase(item);
                     }
                 },
                 [&](Negated const& negated) {
                     // Negated{Joined{}} should be simplified, so don't unmark
                     // them
                     if (!std::get_if<orangeinp::Joined>(&tree_[negated.node]))
                     {
                         remove_orphaned(negated.node);
                     }
                     if (auto item = orphaned_negate_nodes_.find(node_id);
                         item != orphaned_negate_nodes_.end())
                     {
                         orphaned_negate_nodes_.erase(item);
                     }
                 },
                 [](auto&&) {}},
        tree_[node_id]);
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
 * Recusively record that we need to insert \c Negated node for operands of a
 * \c Joined node. This handles cas
 *
 * \param node_id a \c Joined node_id with a \c Negated parent
 */
inline void DeMorganSimplifier::add_new_negated_nodes(NodeId node_id)
{
    CELER_EXPECT(std::get_if<orangeinp::Joined>(&tree_[node_id]));
    auto* join_node = std::get_if<orangeinp::Joined>(&tree_[node_id]);
    if (!join_node)
    {
        CELER_ASSERT_UNREACHABLE();
    }
    for (auto const& join_operand : join_node->nodes)
    {
        std::visit(Overload{[&](Joined const& joined) {
                                // the new Negated node will point to a
                                // Joined node so we need to transform that
                                // Joined node as well
                                add_new_negated_nodes(join_operand);
                                negated_join_nodes_[join_operand] = &joined;
                            },
                            [&](Negated const& negated) {
                                // double negation, this simplifies unless it
                                // has another parent
                                orphaned_negate_nodes_.insert(join_operand);
                                // however, we need to make sure that we're
                                // keeping the target of the double negation
                                // around because someone might need it
                                remove_orphaned(negated.node);
                                // per DeMorgran's law, negate each operand of
                                // the Joined node if we're not inserting a
                                // Negated{Joined{}}.
                                new_negated_nodes_.insert(join_operand);
                            },
                            [&](auto const&) {
                                // per DeMorgran's law, negate each operand of
                                // the Joined node if we're not inserting a
                                // Negated{Joined{}}.
                                new_negated_nodes_.insert(join_operand);
                            }},
                   tree_[join_operand]);
    }
    // assume that the Joined node doesn't have other parents and
    // mark it for deletion. This is done after recusive calls as
    // parents can only have a higher node_id and the recursive
    // calls explore childrens with lower node_id. This can be
    // unmarked later as we process potential parents
    orphaned_join_nodes_.insert(node_id);
}

//---------------------------------------------------------------------------//
/*!
 * Special handling for a \c Joined or \c Negated node. A Joined node can be
 * duplicated if it has multiple parents, e.g. a Negated and a Joined{Negated}
 * parent. Similarly, a Negated node might have to be skipped because it'd only
 * be used in a double negation
 *
 * \return true if an unmodified version of node_id should be inserted in the
 * simplified tree
 */
inline bool DeMorganSimplifier::process_negated_joined_nodes(NodeId node_id,
                                                             CsgTree& result)
{
    return std::visit(
        Overload{
            [&](Negated const&) {
                // This Negated{} node is orphaned, most likely because
                // of a double negation so we don't need to insert it.
                // It still needs to be added in inserted_nodes so that
                // when adding a Joined node, we correctly redirect the
                // operand looking for it to the children
                if (auto orphan_node_id = process_orphaned_negate_node(node_id))
                {
                    auto orphan_node = std::get_if<orangeinp::Negated>(
                        &tree_[orphan_node_id]);
                    if (!orphan_node)
                    {
                        CELER_ASSERT_UNREACHABLE();
                    }
                    inserted_nodes_[node_id]
                        = replace_node_id(orphan_node->node);
                    return false;
                }
                // this node is a Negated{Join} node, we have already
                // inserted the
                // opposite Joined node
                if (auto negated_node = process_transformed_negate_node(node_id))
                {
                    // we don't need to insert the original
                    // Negated{Join} node redirect parents looking for
                    // this node to the new Joined node.
                    inserted_nodes_[node_id]
                        = inserted_nodes_.find(negated_node->node)->second;
                    return false;
                }
                return true;
            },
            [&](Joined const&) {
                // The current node is a Joined{} and we need to insert
                // a Negated version of it.
                if (auto negated_node = process_negated_node(node_id))
                {
                    // Insert the opposite join instead, updating the
                    // children ids.
                    auto const& [op, nodes] = *negated_node;
                    // Lookup the new id of each operand
                    std::vector<NodeId> operands;
                    operands.reserve(nodes.size());
                    for (auto n : nodes)
                    {
                        operands.push_back(replace_node_id(std::move(n)));
                    }

                    auto [new_id, inserted] = result.insert(
                        Joined{(op == logic::land) ? logic::lor : logic::land,
                               std::move(operands)});
                    inserted_nodes_[node_id] = new_id;
                }
                // this Negated{Join} node doesn't have any other
                // parents, so we don't need to insert its original
                // version
                if (process_orphaned_join_node(node_id))
                {
                    return false;
                }
                return true;
            },
            [](auto&&) { return true; },
        },
        tree_[node_id]);
}

//---------------------------------------------------------------------------//
/*!
 * First pass through the tree to find negated set operations
 */
inline void DeMorganSimplifier::find_negated_joins()
{
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        std::visit(
            Overload{
                [&](Negated const& negated) {
                    // we don't need to check if this is a potential parent of
                    // a Joined{} node marked as orphan as this node will also
                    // get simplified.

                    if (std::get_if<orangeinp::Joined>(&tree_[negated.node]))
                    {
                        // This is a Negated{Joined{...}}
                        mark_negated_operator(node_id);
                    }
                },
                [&](Joined const& joined) {
                    for (auto const& join_operand : joined.nodes)
                    {
                        remove_orphaned(join_operand);
                    }
                },
                [&](Aliased const& aliased) { remove_orphaned(aliased.node); },
                [](auto&&) {},
            },
            tree_[node_id]);
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

    // We can now build the new tree.
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        // Negated and Joined nodes need special handling. They can be
        // duplicated or omitted in the simplified tree
        if (!process_negated_joined_nodes(node_id, result))
        {
            continue;
        }

        // We need to insert that node in the simplified
        Node new_node = tree_[node_id];
        // We need to update the childrens' ids in the new tree.
        std::visit(
            Overload{
                [&](Negated& negated) {
                    negated.node = replace_node_id(negated.node);
                },
                [&](Aliased& aliased) {
                    aliased.node = replace_node_id(aliased.node);
                },
                [&](Joined& joined) {
                    for (auto& op : joined.nodes)
                    {
                        // if the node is a Negated{Joined}, we need to match
                        // to a newly inserted node
                        if (auto* negated
                            = std::get_if<orangeinp::Negated>(&tree_[op]);
                            negated
                            && std::get_if<orangeinp::Joined>(
                                &tree_[negated->node]))
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
                },
                [](auto&&) {},
            },
            new_node);

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
    return {};
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
    return {};
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
    return {};
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
    return {};
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
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Utility to check the two maps for the new id of a node.
 * maybe we can coalesce them in a single map<NodeId, vector<NodeId>>
 */
inline NodeId DeMorganSimplifier::replace_node_id(NodeId node_id)
{
    if (auto new_id = inserted_nodes_.find(node_id);
        new_id != inserted_nodes_.end())
    {
        return new_id->second;
    }
    if (auto new_id = original_new_nodes_.find(node_id);
        new_id != original_new_nodes_.end())
    {
        return new_id->second;
    }
    return node_id;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas