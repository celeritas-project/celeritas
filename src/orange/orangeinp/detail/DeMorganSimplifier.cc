//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/DeMorganSimplifier.cc
//---------------------------------------------------------------------------//

#include "DeMorganSimplifier.hh"

#include <unordered_map>
#include <unordered_set>
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
 * Check unmodified, then modified, or default
 */
NodeId
DeMorganSimplifier::MatchingNodes::unmod_mod_or(NodeId default_id) const noexcept
{
    if (unmodified)
        return *unmodified;
    if (modified)
        return *modified;
    return default_id;
}

//---------------------------------------------------------------------------//
/*!
 * Check modified, then unmodified, or default
 */
NodeId
DeMorganSimplifier::MatchingNodes::mod_unmod_or(NodeId default_id) const noexcept
{
    if (modified)
        return *modified;
    if (unmodified)
        return *unmodified;
    return default_id;
}

//---------------------------------------------------------------------------//
/*!
 * Simplify
 */
CsgTree DeMorganSimplifier::operator()()
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
void DeMorganSimplifier::remove_orphaned(NodeId node_id)
{
    std::visit(
        Overload{
            [&](Joined const& joined) {
                // we need to recursively unmark orphaned nodes and unmark
                // ourself
                for (auto join_operand : joined.nodes)
                {
                    remove_orphaned(join_operand);
                }
                if (auto item = orphaned_nodes_.find(node_id);
                    item != orphaned_nodes_.end())
                {
                    CELER_EXPECT(std::holds_alternative<Joined>(tree_[*item]));
                    orphaned_nodes_.erase(item);
                }
            },
            [&](Negated const& negated) {
                // Negated{Joined{}} should be simplified, so don't unmark
                // them
                if (!std::get_if<orangeinp::Joined>(&tree_[negated.node]))
                {
                    remove_orphaned(negated.node);
                }
                if (auto item = orphaned_nodes_.find(node_id);
                    item != orphaned_nodes_.end())
                {
                    CELER_EXPECT(std::holds_alternative<Negated>(tree_[*item]));
                    orphaned_nodes_.erase(item);
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
void DeMorganSimplifier::mark_negated_operator(NodeId negated_id)
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
void DeMorganSimplifier::add_new_negated_nodes(NodeId node_id)
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
                                orphaned_nodes_.insert(join_operand);
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
    orphaned_nodes_.insert(node_id);
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
bool DeMorganSimplifier::process_negated_joined_nodes(NodeId node_id,
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
                if (orphaned_nodes_.extract(node_id))
                {
                    auto orphan_node
                        = std::get_if<orangeinp::Negated>(&tree_[node_id]);
                    if (!orphan_node)
                    {
                        CELER_ASSERT_UNREACHABLE();
                    }
                    auto& matching_nodes = node_ids_translation_[node_id];
                    matching_nodes.modified
                        = node_ids_translation_[orphan_node->node].mod_unmod_or(
                            orphan_node->node);
                    return false;
                }
                // this node is a Negated{Join} node, we have already
                // inserted the
                // opposite Joined node
                if (auto node_handle
                    = transformed_negated_nodes_.extract(node_id))
                {
                    auto& negated_node_child = node_handle.mapped()->node;
                    // we don't need to insert the original
                    // Negated{Join} node redirect parents looking for
                    // this node to the new Joined node.
                    auto& matching_nodes = node_ids_translation_[node_id];
                    matching_nodes.modified
                        = node_ids_translation_[negated_node_child].mod_unmod_or(
                            negated_node_child);
                    return false;
                }
                return true;
            },
            [&](Joined const&) {
                // The current node is a Joined{} and we need to insert
                // a Negated version of it.
                if (auto node_handle = negated_join_nodes_.extract(node_id))
                {
                    // Insert the opposite join instead, updating the
                    // children ids.
                    auto const& [op, nodes] = *node_handle.mapped();
                    // Lookup the new id of each operand
                    std::vector<NodeId> operands;
                    operands.reserve(nodes.size());
                    for (auto n : nodes)
                    {
                        operands.push_back(
                            node_ids_translation_[n].mod_unmod_or(n));
                    }

                    auto [new_id, inserted] = result.insert(
                        Joined{(op == logic::land) ? logic::lor : logic::land,
                               std::move(operands)});
                    node_ids_translation_[node_id].modified = std::move(new_id);
                }
                // this Negated{Join} node doesn't have any other
                // parents, so we don't need to insert its original
                // version if its an orphaned node
                auto node_handle = orphaned_nodes_.extract(node_id);
                return !(node_handle && node_handle.value());
            },
            [](auto&&) { return true; },
        },
        tree_[node_id]);
}

//---------------------------------------------------------------------------//
/*!
 * First pass through the tree to find negated set operations
 */
void DeMorganSimplifier::find_negated_joins()
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

    // volume nodes act as tags on a NodeId indicating that it is the root of a
    // volume they need to be kept
    std::unordered_set<NodeId> volume_roots{tree_.volumes().cbegin(),
                                            tree_.volumes().cend()};
    for (auto node_id : volume_roots)
    {
        if (auto iter = orphaned_nodes_.find(node_id);
            iter != orphaned_nodes_.end())
        {
            remove_orphaned(*iter);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Second pass through the tree to build the simplified tree.
 *
 * \return the simplified tree
 */
CsgTree DeMorganSimplifier::build_simplified_tree()
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
                    negated.node
                        = node_ids_translation_[negated.node].mod_unmod_or(
                            negated.node);
                },
                [&](Aliased& aliased) {
                    aliased.node
                        = node_ids_translation_[aliased.node].mod_unmod_or(
                            aliased.node);
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
                            op = node_ids_translation_[op].mod_unmod_or(op);
                        }
                        // otherwise, only search unmodified nodes
                        else if (auto& new_id = node_ids_translation_[op];
                                 new_id.unmodified)
                        {
                            op = *new_id.unmodified;
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
        node_ids_translation_[node_id].unmodified = new_id;

        // We might have to insert a negated version of that node
        if (new_negated_nodes_.extract(node_id))
        {
            auto [new_negated_node_id, negated_inserted]
                = result.insert(Negated{new_id});
            node_ids_translation_[node_id].modified = new_negated_node_id;
        }
    }

    // set the volumes in the simplified tree
    for (auto volume : tree_.volumes())
    {
        // volumes should be kept, so we must have a equivalent node in the new
        // tree or in the replaced tree (if the volume was pointing to a
        // negated Join)
        CELER_EXPECT(node_ids_translation_[volume]);

        result.insert_volume(
            node_ids_translation_[volume].unmod_mod_or(volume));
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
