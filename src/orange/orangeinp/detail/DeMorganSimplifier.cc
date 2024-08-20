//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/DeMorganSimplifier.cc
//---------------------------------------------------------------------------//

#include "DeMorganSimplifier.hh"

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
 * Construct and fix bitset size.
 */
DeMorganSimplifier::DeMorganSimplifier(CsgTree const& tree) : tree_(tree)
{
    parents_of.resize(tree_.size());
    new_negated_nodes_.resize(tree_.size());
    negated_join_nodes_.resize(tree_.size());
    node_ids_translation_.resize(tree_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Perform the simplification. The state of the instance isn't cleared, so only
 * call this once.
 */
CsgTree DeMorganSimplifier::operator()()
{
    this->find_join_negations();
    return this->build_simplified_tree();
}

//---------------------------------------------------------------------------//
/*!
 * First pass through the tree to find negated set operations.
 */
void DeMorganSimplifier::find_join_negations()
{
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        std::visit(
            Overload{
                [&](Negated const& negated) {
                    parents_of[negated.node.get()].insert(node_id);
                    if (std::holds_alternative<Joined>(tree_[negated.node]))
                    {
                        // This is a Negated{Joined{...}}
                        negated_join_nodes_[negated.node.get()] = true;
                        this->add_negation_for_operands(negated.node);
                    }
                },
                [&](Joined const& joined) {
                    // we found a new parent for each operand
                    for (auto const& join_operand : joined.nodes)
                    {
                        parents_of[join_operand.get()].insert(node_id);
                    }
                },
                [&](Aliased const&) {
                    // not supported
                    CELER_ASSERT_UNREACHABLE();
                },
                // nothing to do for leaf node types
                [](auto&&) {},
            },
            tree_[node_id]);
    }

    // Volume nodes act as tags on a NodeId indicating that it is the root of a
    // volume, so these subtrees need to be preserved.
    // Consider a "virtual" parent for these nodes
    for (auto node_id : tree_.volumes())
    {
        parents_of[node_id.get()].insert(NodeId{});
    }
}

//---------------------------------------------------------------------------//
/*!
 * Recursively record that we need to insert \c Negated node for operands of a
 * \c Joined node.
 *
 * \param node_id a \c Joined node_id with a \c Negated parent.
 */
void DeMorganSimplifier::add_negation_for_operands(NodeId node_id)
{
    CELER_ASSUME(std::holds_alternative<Joined>(tree_[node_id]));
    auto& [op, operands] = std::get<Joined>(tree_[node_id]);

    for (auto const& join_operand : operands)
    {
        if (std::holds_alternative<Joined>(tree_[join_operand]))
        {
            // The operand is a Joined node, and we're
            // about to insert a new Negated node pointing
            // to a Joined node.
            // So we transform that Joined node as well,
            // and we skip the insertion of a Negated node
            // pointing to that operand
            negated_join_nodes_[join_operand.get()] = true;
            // we still have to recursively check
            // descendent
            this->add_negation_for_operands(join_operand);
        }
        else if (!std::holds_alternative<Negated>(tree_[join_operand]))
        {
            // per DeMorgran's law, negate each operand of
            // the Joined node if we're not inserting a
            // Negated{Joined{}}.
            new_negated_nodes_[join_operand.get()] = true;
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
        if (!this->process_negated_joined_nodes(node_id, result))
        {
            continue;
        }

        // We need to insert that node in the simplified tree so make a copy
        Node new_node = tree_[node_id];
        // We need to update the children's ids in the new tree.

        std::visit(
            Overload{
                [&](Negated& negated) {
                    // we're inserting a Negated node, it has to point to a
                    // unmodified node negated join would have been simplified
                    // and double negation are not inserted
                    CELER_EXPECT(
                        node_ids_translation_[negated.node.get()].unmodified);
                    negated.node
                        = node_ids_translation_[negated.node.get()].unmodified;
                },
                [&](Joined& joined) {
                    // update each operand of the joined node, it can't
                    // have a negation as parent, or it would have been
                    // inserted in process_negated_joined_nodes
                    for (auto& op : joined.nodes)
                    {
                        // if the node has been simplified, insert the
                        // simplification, otherwise retrieve the new id
                        // of the same node
                        auto& trans = node_ids_translation_[op.get()];
                        if (trans.simplified_to)
                        {
                            op = trans.simplified_to;
                        }
                        else
                        {
                            CELER_EXPECT(trans.unmodified);
                            op = trans.unmodified;
                        }
                    }
                },
                // other nodes don't have children
                [](auto&&) {},
            },
            new_node);

        auto [new_id, inserted] = result.insert(std::move(new_node));
        // Record the new node id
        auto& trans = node_ids_translation_[node_id.get()];
        CELER_EXPECT(!trans.unmodified);
        trans.unmodified = new_id;

        // We might have to insert a negated version of that node
        if (new_negated_nodes_[node_id.get()])
        {
            auto [new_negated_node_id, negated_inserted]
                = result.insert(Negated{new_id});
            CELER_EXPECT(!trans.new_negation);
            trans.new_negation = new_negated_node_id;
        }
    }

    // set the volumes in the simplified tree by checking the translation map
    for (auto volume : tree_.volumes())
    {
        // Volumes should be kept, so we must have an equivalent node in the
        // new tree.
        // This is not always the exact same node, e.g., if the volume
        // points to a negated join, it will still be simplified
        auto& trans = node_ids_translation_[volume.get()];

        // if the node has been simplified, insert the simplification,
        // otherwise retrieve the new id of the same node
        if (trans.simplified_to)
        {
            result.insert_volume(trans.simplified_to);
        }
        else
        {
            CELER_EXPECT(trans.unmodified);
            result.insert_volume(trans.unmodified);
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Special handling for a \c Joined or \c Negated node. A Joined node can be
 * duplicated if it has multiple parents, e.g., a Negated and a Joined{Negated}
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
            [&](Negated const& negated) {
                // this node has a joined child, we can never insert it in the
                // simplified tree
                if (std::holds_alternative<Joined>(tree_[negated.node]))
                {
                    // redirect parents looking for this node to the new Joined
                    // node.
                    CELER_EXPECT(
                        node_ids_translation_[negated.node.get()].opposite_join);
                    node_ids_translation_[node_id.get()].simplified_to
                        = node_ids_translation_[negated.node.get()].opposite_join;
                    return false;
                }

                auto& parents = parents_of[node_id.get()];
                // root node, we must insert it
                if (parents.empty())
                {
                    return true;
                }
                // if the only parents of this node are
                // 1. Negated node
                // 2. Join node that have a negated parent, i.e., the Joined
                // parent would change to a Negated parent in the simplified
                // tree then we don't need to insert that negation and only the
                // opposite join is enough
                // --> if this node has at least one volume or non-negated join
                // parent, we need to insert is
                for (auto p : parents)
                {
                    // !NodeId{} is used when a volume is a parent, we need to
                    // insert that node
                    if (!p
                        || (std::holds_alternative<Joined>(tree_[p])
                            && this->should_insert_join(std::move(p))))
                    {
                        return true;
                    }
                }
                // this node isn't inserted in the simplified tree because
                // it would only have negated parents, redirect them to
                // the target of the double negation
                CELER_EXPECT(
                    node_ids_translation_[negated.node.get()].unmodified);
                node_ids_translation_[node_id.get()].double_negation_target
                    = node_ids_translation_[negated.node.get()].unmodified;
                return false;
            },
            [&](Joined const& joined) {
                // The current node is a Joined node, and we need to insert
                // a Negated version of it.
                if (negated_join_nodes_[node_id.get()])
                {
                    // Insert the opposite join instead, updating the
                    // children ids.
                    auto const& [op, nodes] = joined;
                    // Lookup the new id of each operand
                    std::vector<NodeId> operands;
                    operands.reserve(nodes.size());
                    for (auto n : nodes)
                    {
                        // we're adding a negated join with a negated children,
                        // simplify by pointing to the children of the negation
                        if (auto* neg = std::get_if<Negated>(&tree_[n]))
                        {
                            operands.push_back([&] {
                                CELER_EXPECT(
                                    node_ids_translation_[n.get()]
                                        .double_negation_target
                                    || node_ids_translation_[neg->node.get()]
                                           .unmodified);
                                // the negated node only has double negation,
                                // so it no longer exists in the result tree
                                // get the target of the double negation
                                if (node_ids_translation_[n.get()]
                                        .double_negation_target)
                                    return node_ids_translation_[n.get()]
                                        .double_negation_target;

                                // the negated node still exists because
                                // another node refers to it, redirect to its
                                // child.
                                return node_ids_translation_[neg->node.get()]
                                    .unmodified;
                            }());
                        }
                        else
                        {
                            // otherwise, we should have inserted a negated
                            // version of it or simplified it with DeMorgan
                            operands.push_back([&] {
                                auto& trans = node_ids_translation_[n.get()];
                                CELER_EXPECT(trans.new_negation
                                             || trans.opposite_join);
                                if (trans.new_negation)
                                    return trans.new_negation;
                                return trans.opposite_join;
                            }());
                        }
                    }

                    auto [new_id, inserted] = result.insert(
                        Joined{(op == logic::land) ? logic::lor : logic::land,
                               std::move(operands)});
                    node_ids_translation_[node_id.get()].opposite_join
                        = std::move(new_id);
                }
                return this->should_insert_join(node_id);
            },
            // other nodes need to be inserted
            [](auto&&) { return true; },
        },
        tree_[node_id]);
}

//---------------------------------------------------------------------------//
/*!
 * Determine if the \c Joined node referred by node_id must be inserted in the
 * simplified tree.
 *
 * \param node_id the \c Joined node to process.
 *
 * \return true if an equivalent of this join node must be inserted
 */
bool DeMorganSimplifier::should_insert_join(NodeId node_id)
{
    CELER_EXPECT(std::holds_alternative<Joined>(tree_[node_id]));
    auto& parents = parents_of[node_id.get()];
    // root node, we must insert it
    if (parents.empty())
    {
        return true;
    }
    // We must insert the original join node if one of the following is true
    // 1. It is pointed to directly by a volume
    // 2. It has a Join ancestor that is not negated
    // 3. It has a negated parent and that negated node has a negated join
    // parent (double negation of that join)
    auto has_negated_join_parent = [&](NodeId n) {
        for (auto p : parents_of[n.get()])
        {
            if (p && negated_join_nodes_[p.get()])
                return true;
        }
        return false;
    };
    for (auto p : parents)
    {
        // !NodeId{} is used when a volume is a parent, we need to insert that
        // node
        // TODO: Is it really correct in all cases...
        if (!p
            || (std::holds_alternative<Joined>(tree_[p])
                && this->should_insert_join(p))
            || (std::holds_alternative<Negated>(tree_[p])
                && has_negated_join_parent(p)))
        {
            return true;
        }
    }
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
