//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeUtils.cc
//---------------------------------------------------------------------------//
#include "CsgTreeUtils.hh"

#include <algorithm>
#include <variant>
#include <vector>

#include "corecel/cont/Range.hh"

#include "detail/DeMorganSimplifier.hh"
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
            auto updated = state[n.get()];
            if ((updated == NodeReplacer::known_true
                 || updated == NodeReplacer::known_false)
                && std::holds_alternative<Surface>((*tree)[n]))
            {
                max_node = std::max(max_node, n);
                tree->exchange(n,
                               updated == NodeReplacer::known_true
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
TransformedTree transform_negated_joins(CsgTree const& tree)
{
    return detail::DeMorganSimplifier{tree}();
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
