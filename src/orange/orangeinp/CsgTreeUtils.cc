//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeUtils.cc
//---------------------------------------------------------------------------//
#include "CsgTreeUtils.hh"

#include <algorithm>
#include <utility>
#include <variant>
#include <vector>

#include "corecel/cont/Range.hh"

#include "detail/InfixStringBuilder.hh"
#include "detail/NodeReplacementInserter.hh"
#include "detail/PostfixLogicBuilder.hh"

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
 * \return Node ID of the lowest node that required simplification.
 */
NodeId replace_down(CsgTree* tree, NodeId n, Node repl)
{
    CELER_EXPECT(is_boolean_node(repl));

    detail::NodeReplacementInserter::VecNode stack{{n, std::move(repl)}};

    NodeId lowest_node{n};

    while (!stack.empty())
    {
        n = std::move(stack.back().first);
        repl = std::move(stack.back().second);
        stack.pop_back();
        lowest_node = std::min(n, lowest_node);

        Node prev = tree->exchange(n, std::move(repl));
        std::visit(detail::NodeReplacementInserter{&stack, repl}, prev);
    }
    return lowest_node;
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
        bool simplified = tree->simplify(node_id);
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
 * Convert a node to postfix notation.
 */
[[nodiscard]] std::vector<LocalSurfaceId::size_type>
build_postfix(CsgTree const& tree, NodeId n)
{
    CELER_EXPECT(n < tree.size());
    std::vector<LocalSurfaceId::size_type> result;
    detail::PostfixLogicBuilder build_impl{tree, &result};

    build_impl(n);
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
