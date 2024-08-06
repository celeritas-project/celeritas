//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeUtils.cc
//---------------------------------------------------------------------------//
#include "CsgTreeUtils.hh"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTypes.hh"

#include "detail/InfixStringBuilder.hh"
#include "detail/NodeReplacer.hh"

namespace celeritas
{
namespace orangeinp
{

//---------------------------------------------------------------------------//
/*!
 * Replace the given node ID with the replacement node.
 *
 * This recurses through daughters of "Joined" to simplify their originating
 * surfaces if possible.
 * - "negated": non-constant daughter node is replaced with ~b
 * - "replaced": non-constant daughter node is replaced with `b`
 * - "joined": for (false, or): all daughters are "false"
 *             for (true, and): all daughters are "true"
 * - surface: "true"
 * - constant: check for contradiction
 *
 * This operation is at worst O((number of nodes) * (depth of graph)).
 */
std::vector<NodeId>
replace_and_simplify(CsgTree* tree, NodeId repl_key, Node repl_value)
{
    CELER_EXPECT(tree);
    CELER_EXPECT(repl_key < tree->size());
    CELER_ASSUME(is_boolean_node(repl_value));

    using detail::NodeReplacer;

    NodeId max_node{repl_key};
    NodeReplacer::VecRepl state{tree->size(), NodeReplacer::unvisited};
    state[CsgTree::true_node_id().get()] = NodeReplacer::known_true;
    state[CsgTree::false_node_id().get()] = NodeReplacer::known_false;

    state[repl_key.get()] = (std::holds_alternative<True>(repl_value)
                                 ? NodeReplacer::known_true
                                 : NodeReplacer::known_false);

    bool simplifying{true};
    do
    {
        // Sweep backward, updating known state
        simplifying = false;
        for (auto n = max_node; n > CsgTree::false_node_id(); --n)
        {
            bool updated
                = std::visit(detail::NodeReplacer{&state, n}, (*tree)[n]);
            simplifying = simplifying || updated;
        }

        // Replace literals and simplify
        for (auto n : range(CsgTree::false_node_id() + 1, NodeId{tree->size()}))
        {
            auto repl_value = state[n.get()];
            if ((repl_value == NodeReplacer::known_true
                 || repl_value == NodeReplacer::known_false)
                && std::holds_alternative<Surface>((*tree)[n]))
            {
                max_node = std::max(max_node, n);
                tree->exchange(n,
                               repl_value == NodeReplacer::known_true
                                   ? Node{True{}}
                                   : Node{False{}});
            }
            else if (auto simplified = tree->simplify(n))
            {
                max_node = std::max(max_node, n);
                simplifying = true;
            }
        }
    } while (simplifying);

    std::vector<NodeId> unknown_surface_nodes;

    // Replace nonliterals
    for (auto n : range(CsgTree::false_node_id() + 1, NodeId{tree->size()}))
    {
        auto repl_state = state[n.get()];
        if (std::holds_alternative<Surface>((*tree)[n]))
        {
            if (repl_state == NodeReplacer::unknown)
            {
                // Keep track of boundary surfaces that we can't prove, likely
                // because of a union boundary
                unknown_surface_nodes.push_back(n);
            }
        }
        else if (repl_state == NodeReplacer::known_true)
        {
            tree->exchange(n, Node{True{}});
        }
        else if (repl_state == NodeReplacer::known_false)
        {
            tree->exchange(n, Node{False{}});
        }
        else
        {
            tree->simplify(n);
        }
    }
    return unknown_surface_nodes;
}

//---------------------------------------------------------------------------//
/*!
 * Simplify all nodes in the tree starting with this one.
 *
 * \return Lowest ID of any simplified node
 */
NodeId simplify_up(CsgTree* tree, NodeId start)
{
    CELER_EXPECT(tree);
    CELER_EXPECT(start < tree->size());

    // Sweep bottom to top to simplify the tree
    NodeId result;
    for (auto node_id : range(start, NodeId{tree->size()}))
    {
        auto simplified = tree->simplify(node_id);
        if (simplified && !result)
        {
            result = node_id;
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Iteratively simplify all nodes in the tree.
 *
 * The input 'start' node should be the minimum node from a \c replace_down
 * operation. In the worst case, it should take as many sweeps as the depth of
 * the tree.
 */
void simplify(CsgTree* tree, NodeId start)
{
    CELER_EXPECT(tree);
    CELER_EXPECT(start > tree->false_node_id() && start < tree->size());

    while (start)
    {
        auto next_start = simplify_up(tree, start);
        CELER_ASSERT(!next_start || next_start > start);
        start = next_start;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Simplify negated joins using De Morgan's law.
 *
 * This is required if the tree's logic expression is used with
 * \c InfixEvaluator as negated joins are not supported.
 */
CsgTree transform_negated_joins(CsgTree const& tree)
{
    // For each node_id in that set, we'll create a Negated{node_id} node in
    // the new tree.
    std::unordered_set<NodeId> new_negated_nodes;

    // We need to insert a negated version of these Joined nodes.
    // The value is the Node referred to by the key as to avoid an extra
    // get_if.
    std::unordered_map<NodeId, Joined const*> negated_join_nodes;

    // Contains the node_id of Negated{Joined}. These don't need to be inserted
    // in the new tree and its parent can directly point to the transformed
    // Joined node.
    std::unordered_map<NodeId, Negated const*> transformed_negated_nodes;

    // A Negated{Joined{}} nodes needs to be mapped to the opposite Joined{}
    // node. Parents need to point to the new Joined{} node. The old Joined{}
    // node needs to be kept if it had parents other than the Negated node.
    std::unordered_set<NodeId> orphaned_join_nodes;

    // Recursive lambda. Negated{node_id} should be added to the new tree.
    // if node_id is a Joined node, recusively add its children as negated
    // nodes.
    // TODO: extract the lambda to a free function
    auto add_new_negated_nodes = [&](NodeId node_id) {
        auto impl = [&](auto const& self, NodeId node_id) -> void {
            CELER_EXPECT(std::get_if<orangeinp::Joined>(&tree[node_id]));
            if (auto* join_node = std::get_if<orangeinp::Joined>(&tree[node_id]))
            {
                for (auto const& join_operand : join_node->nodes)
                {
                    if (auto* joined_join_operand
                        = std::get_if<orangeinp::Joined>(&tree[join_operand]))
                    {
                        // the new Negated node will point to a Joined node
                        // so we need to transform that Joined node as well
                        self(self, join_operand);
                        negated_join_nodes[join_operand] = joined_join_operand;
                    }
                    // per DeMorgran's law, negate each operand of the Joined
                    // node.
                    new_negated_nodes.insert(join_operand);
                }
                // assume that the Joined node doesn't have other parents and
                // mark it for deletion. This is done after recusive calls as
                // parents can only have a higher node_id and the recursive
                // calls explore childrens with lower node_id. This can be
                // unmarked later as we process potential parents
                orphaned_join_nodes.insert(node_id);
            }
        };

        impl(impl, std::move(node_id));
    };

    // check if we can unmark a node marked for deletion
    auto remove_orphaned = [&](NodeId node_id) {
        if (auto item = orphaned_join_nodes.find(node_id);
            item != orphaned_join_nodes.end())
        {
            orphaned_join_nodes.erase(item);
        }
    };

    // First pass through all nodes to find all nand / nor
    for (auto node_id : range(NodeId{tree.size()}))
    {
        // TODO: visitor
        if (auto* node_ptr = &tree[node_id];
            auto* negated = std::get_if<orangeinp::Negated>(node_ptr))
        {
            // we don't need to check if this is a potential parent of a
            // Joined{} node marked as orphan as this node will also get
            // simplified.
            if (auto* joined
                = std::get_if<orangeinp::Joined>(&tree[negated->node]))
            {
                // This is a Negated{Joined{...}}
                // we'll need to add a Negated{} for each of its children.
                add_new_negated_nodes(negated->node);
                negated_join_nodes[negated->node] = joined;
                transformed_negated_nodes[node_id] = negated;
            }
        }
        else if (auto* aliased = std::get_if<orangeinp::Aliased>(node_ptr))
        {
            remove_orphaned(aliased->node);  // TODO: handle alias pointing to
                                             // an alias (pointing to an
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

    CsgTree result{};

    // for a given node_id in the original tree, map to its id in the new tree
    std::unordered_map<NodeId, NodeId> inserted_nodes;

    // A node in the original tree can map to more than one node in the new
    // tree
    std::unordered_map<NodeId, NodeId> original_new_nodes;

    // We can now build the new tree.
    for (auto node_id : range(NodeId{tree.size()}))
    {
        // The current node is a Joined{} and we need to insert a Negated
        // version of it.
        if (auto iter = negated_join_nodes.find(node_id);
            iter != negated_join_nodes.end())
        {
            // Insert the opposite join instead, updating the children ids.
            Joined const& join = *iter->second;
            OperatorToken opposite_op = (join.op == logic::land)
                                            ? logic::lor
                                            : logic::OperatorToken::land;
            // Lookup the new id of each operand
            std::vector<NodeId> operands;
            operands.reserve(join.nodes.size());
            for (auto operand : join.nodes)
            {
                if (auto new_id = inserted_nodes.find(operand);
                    new_id != inserted_nodes.end())
                {
                    operands.push_back(new_id->second);
                }
            }
            auto [new_id, inserted]
                = result.insert(Joined{opposite_op, std::move(operands)});
            inserted_nodes[node_id] = new_id;
            negated_join_nodes.erase(iter);
        }

        // this Negated{Join} node doesn't have any other parents, so we don't
        // need to insert its original version
        if (auto iter = orphaned_join_nodes.find(node_id);
            iter != orphaned_join_nodes.end())
        {
            orphaned_join_nodes.erase(iter);
            continue;
        }

        // this node is a Negated{Join} node, we have already inserted the
        // opposite Joined node
        if (auto negated_node = transformed_negated_nodes.find(node_id);
            negated_node != transformed_negated_nodes.end())
        {
            // we don't need to insert the original Negated{Join} node
            transformed_negated_nodes.erase(negated_node);
            // redirect parents looking for this node to the new Joined node.
            inserted_nodes[node_id]
                = inserted_nodes.find(negated_node->second->node)->second;
            continue;
        }

        // Utility to check the two maps for the new id of a node.
        // maybe we can coalesce them in a single map<NodeId, vector<NodeId>>
        auto replace_node_id = [&](NodeId n) {
            if (auto new_id = inserted_nodes.find(n);
                new_id != inserted_nodes.end())
            {
                return new_id->second;
            }
            if (auto new_id = original_new_nodes.find(n);
                new_id != inserted_nodes.end())
            {
                return new_id->second;
            }
            return n;
        };
        // this node isn't a transformed Join or Negated node, so we can insert
        // it.
        Node new_node = tree[node_id];

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
                op = replace_node_id(op);
            }
        }
        auto [new_id, inserted] = result.insert(std::move(new_node));
        // this is recorded in a different map as a node in the original tree
        // can be inserted multiple times in the new tree e.g. a
        // Negated{Joined}, if that Joined node has another parent.
        original_new_nodes[node_id] = new_id;

        // We might have to insert a negated version of that node
        if (auto iter = new_negated_nodes.find(node_id);
            iter != new_negated_nodes.end())
        {
            auto [new_negated_node_id, negated_inserted]
                = result.insert(Negated{new_id});
            inserted_nodes[node_id] = new_negated_node_id;
            new_negated_nodes.erase(iter);
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert a node to an infix string expression.
 */
[[nodiscard]] std::string build_infix_string(CsgTree const& tree, NodeId n)
{
    CELER_EXPECT(n < tree.size());
    std::ostringstream os;
    detail::InfixStringBuilder build_impl{tree, &os};

    build_impl(n);
    return os.str();
}

//---------------------------------------------------------------------------//
/*!
 * Construct the sorted set of all surfaces that are part of the tree.
 *
 * This list removes surfaces that have been eliminated by logical replacement.
 * Thanks to the CSG tree's deduplication, each surface should appear in the
 * tree at most once.
 */
[[nodiscard]] std::vector<LocalSurfaceId> calc_surfaces(CsgTree const& tree)
{
    std::vector<LocalSurfaceId> result;
    for (auto node_id : range(NodeId{tree.size()}))
    {
        if (Surface const* s = std::get_if<Surface>(&tree[node_id]))
        {
            result.push_back(s->id);
        }
    }
    std::sort(result.begin(), result.end());
    CELER_ENSURE(std::unique(result.begin(), result.end()) == result.end());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
