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
 * Check unmodified, then modified, or default.
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
 * Check modified, then unmodified, or default.
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
 * Construct and fix bitset size.
 */
DeMorganSimplifier::DeMorganSimplifier(CsgTree const& tree) : tree_(tree)
{
    orphaned_nodes_.resize(tree_.size());
    new_negated_nodes_.resize(tree_.size());
    simplified_negated_nodes_.resize(tree_.size());
    negated_join_nodes_.resize(tree_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Perform the simplification. The state of the instance isn't cleared, so only
 * call this once.
 */
CsgTree DeMorganSimplifier::operator()()
{
    find_join_negations();
    return build_simplified_tree();
}

//---------------------------------------------------------------------------//
/*!
 * During the first pass, a negated join operation is by default flag for
 * removal during the construction of the simplified tree. If we later discover
 * another parent that needs to keep the \c Joined node this function remove
 * the \c Joined node and its descendents from the list of orphans.
 *
 * \param node_id the node to keep and its descendents.
 */
void DeMorganSimplifier::record_parent_for(NodeId node_id)
{
    std::visit(Overload{[&](Joined const& joined) {
                            // each operand of the Joined node also needs to be
                            // kept
                            for (auto join_operand : joined.nodes)
                            {
                                record_parent_for(join_operand);
                            }
                            // and the Joined node itself needs to be kept
                            if (auto ref = orphaned_nodes_[node_id.get()])
                            {
                                CELER_EXPECT(std::holds_alternative<Joined>(
                                    tree_[node_id]));
                                ref = false;
                            }
                        },
                        [&](Negated const& negated) {
                            // Negated{Joined{}} should be simplified, so don't
                            // unmark them
                            if (!std::get_if<Joined>(&tree_[negated.node]))
                            {
                                record_parent_for(negated.node);
                            }

                            if (auto ref = orphaned_nodes_[node_id.get()])
                            {
                                CELER_EXPECT(std::holds_alternative<Negated>(
                                    tree_[node_id]));
                                ref = false;
                            }
                        },
                        [](auto&&) {}},
               tree_[node_id]);
}

//---------------------------------------------------------------------------//
/*!
 * Declare a \c Negated node with a \c Joined child that must be transformed to
 * the opposite join operation. Per DeMorgan's law, this will also make sure
 * that all operands of the join operation have a \c Negated parent added.
 *
 * \param negated_id of a \c Negated node pointing to \c Joined node inside
 * tree_.
 */
void DeMorganSimplifier::record_join_negation(NodeId negated_id)
{
    CELER_EXPECT(negated_id < tree_.size());

    CELER_ASSERT(std::holds_alternative<Negated>(tree_[negated_id]));
    auto* negated_node = std::get_if<Negated>(&tree_[negated_id]);

    simplified_negated_nodes_[negated_id.get()] = true;

    CELER_ASSERT(std::holds_alternative<Joined>(tree_[negated_node->node]));

    negated_join_nodes_[negated_node->node.get()] = true;
    add_negation_for_operands(negated_node->node);
}

//---------------------------------------------------------------------------//
/*!
 * Recusively record that we need to insert \c Negated node for operands of a
 * \c Joined node.
 *
 * \param node_id a \c Joined node_id with a \c Negated parent.
 */
void DeMorganSimplifier::add_negation_for_operands(NodeId node_id)
{
    CELER_EXPECT(std::holds_alternative<Joined>(tree_[node_id]));
    auto* join_node = std::get_if<Joined>(&tree_[node_id]);
    if (!join_node)
    {
        CELER_ASSERT_UNREACHABLE();
    }
    for (auto const& join_operand : join_node->nodes)
    {
        std::visit(Overload{[&](Joined const&) {
                                // the operand is a Joined node and we're about
                                // to insert a new Negated node pointing to a
                                // Joined node. So we transform that
                                // Joined node as well and we skip the
                                // insertion of a Negated node pointing to that
                                // operand
                                negated_join_nodes_[join_operand.get()] = true;
                                // we still have to recusively check descendent
                                add_negation_for_operands(join_operand);
                            },
                            [&](Negated const& negated) {
                                // this would be a double negation, they
                                // self-destruct so mark the operand as an
                                // orphan so that it gets removed
                                orphaned_nodes_[join_operand.get()] = true;
                                // however, we need to make sure that we're
                                // keeping the target of the double negation
                                // because the join operation will need it
                                record_parent_for(negated.node);
                                // per DeMorgran's law, negate each operand of
                                // the Joined node if we're not inserting a
                                // Negated{Joined{}}.
                                // we still need to try to insert the double
                                // negation, it will get simplified during
                                // construction of the simplified tree to
                                // redirect to the new node id
                                new_negated_nodes_[join_operand.get()] = true;
                            },
                            [&](auto const&) {
                                // per DeMorgran's law, negate each operand of
                                // the Joined node if we're not inserting a
                                // Negated{Joined{}}.
                                new_negated_nodes_[join_operand.get()] = true;
                            }},
                   tree_[join_operand]);
    }
    // assume that the Joined node doesn't have other parents and  mark it for
    // deletion.
    orphaned_nodes_[node_id.get()] = true;
}

//---------------------------------------------------------------------------//
/*!
 * Special handling for a \c Joined or \c Negated node. A Joined node can be
 * duplicated if it has multiple parents, e.g. a Negated and a Joined{Negated}
 * parent. Similarly, a Negated node might have to be skipped because it'd only
 * be used in a double negation.
 *
 * \param node_id the \c Negated or \c Joined node to process.
 * \param result the simplified tree being built.
 *
 * \return true if an unmodified version of node_id should be inserted in the
 * simplified tree.
 */
bool DeMorganSimplifier::process_negated_joined_nodes(NodeId node_id,
                                                      CsgTree& result)
{
    return std::visit(
        Overload{
            [&](Negated const&) {
                // This Negated{} node is orphaned, because
                // of a double negation so we don't need to insert it and
                // redirect to its child. It still needs to be added as a
                // modified node so that when adding a Joined node, we
                // correctly redirect the operand looking for it to the
                // children
                if (orphaned_nodes_[node_id.get()])
                {
                    auto orphan_node = std::get_if<Negated>(&tree_[node_id]);
                    if (!orphan_node)
                    {
                        CELER_ASSERT_UNREACHABLE();
                    }
                    node_ids_translation_[node_id].modified
                        = node_ids_translation_[orphan_node->node].mod_unmod_or(
                            orphan_node->node);
                    return false;
                }
                // this node is a Negated node with a join child, we have
                // already inserted the opposite Joined node
                if (simplified_negated_nodes_[node_id.get()])
                {
                    CELER_EXPECT(
                        std::holds_alternative<Negated>(tree_[node_id]));

                    auto& negated_node_child
                        = std::get_if<Negated>(&tree_[node_id])->node;
                    // we don't need to insert the Negated node, simply
                    // redirect parents looking for this node to the new Joined
                    // node.
                    auto& matching_nodes = node_ids_translation_[node_id];
                    matching_nodes.modified
                        = node_ids_translation_[negated_node_child].mod_unmod_or(
                            negated_node_child);
                    return false;
                }
                // not a double negation, not a join negation so we must insert
                // that node
                return true;
            },
            [&](Joined const&) {
                // The current node is a Joined node and we need to insert
                // a Negated version of it.
                if (negated_join_nodes_[node_id.get()])
                {
                    CELER_EXPECT(
                        std::holds_alternative<Joined>(tree_[node_id]));
                    // Insert the opposite join instead, updating the
                    // children ids.
                    auto const& [op, nodes]
                        = *std::get_if<Joined>(&tree_[node_id]);
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
                // if this joined node doesn't have any
                // parents other than the simplified negation, we don't need to
                // insert
                return !(orphaned_nodes_[node_id.get()]);
            },
            // other nodes need to be inserted
            [](auto&&) { return true; },
        },
        tree_[node_id]);
}

//---------------------------------------------------------------------------//
/*!
 * First pass through the tree to find negated set operations.
 */
void DeMorganSimplifier::find_join_negations()
{
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        std::visit(Overload{
                       [&](Negated const& negated) {
                           // record_parents unmark a Joined or Negated node
                           // previously marked as orphan so it doesn't need to
                           // be called here because a child of a Negated node
                           // always simplify( double negation or negated join)
                           if (std::get_if<Joined>(&tree_[negated.node]))
                           {
                               // This is a Negated{Joined{...}}
                               record_join_negation(node_id);
                           }
                       },
                       [&](Joined const& joined) {
                           // we found a new parent for each operands
                           for (auto const& join_operand : joined.nodes)
                           {
                               record_parent_for(join_operand);
                           }
                       },
                       [&](Aliased const& aliased) {
                           record_parent_for(aliased.node);
                       },
                       [](auto&&) {},
                   },
                   tree_[node_id]);
    }

    // volume nodes act as tags on a NodeId indicating that it is the root of a
    // volume so these subtrees need to be preserved. Consider a "virtual"
    // parent for these nodes
    std::unordered_set<NodeId> volume_roots{tree_.volumes().cbegin(),
                                            tree_.volumes().cend()};
    for (auto node_id : volume_roots)
    {
        if (orphaned_nodes_[node_id.get()])
        {
            record_parent_for(node_id);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Second pass through the tree to build the simplified tree.
 *
 * \return the simplified tree.
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

        // We need to insert that node in the simplified tree so make a copy
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
                    // update each operand of the joined node
                    // this joined can't have a negation as parent, it would
                    // have be inserted in process_negated_joined_nodes
                    for (auto& op : joined.nodes)
                    {
                        op = node_ids_translation_[op].unmod_mod_or(op);
                    }
                },
                [](auto&&) {},
            },
            new_node);

        auto [new_id, inserted] = result.insert(std::move(new_node));
        // Record the new node id
        CELER_EXPECT(!node_ids_translation_[node_id].unmodified);
        node_ids_translation_[node_id].unmodified = new_id;

        // We might have to insert a negated version of that node
        if (new_negated_nodes_[node_id.get()])
        {
            auto [new_negated_node_id, negated_inserted]
                = result.insert(Negated{new_id});
            CELER_EXPECT(!node_ids_translation_[node_id].modified);
            node_ids_translation_[node_id].modified = new_negated_node_id;
        }
    }

    // set the volumes in the simplified tree by checking the translation map
    for (auto volume : tree_.volumes())
    {
        // volumes should be kept, so we must have a equivalent node in the new
        // tree. This is not always the exact same node, e.g. if the volume
        // points to a negated join, it will still be simplified
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
