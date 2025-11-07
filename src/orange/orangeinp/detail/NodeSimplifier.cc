//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/NodeSimplifier.cc
//---------------------------------------------------------------------------//
#include "NodeSimplifier.hh"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "corecel/math/Algorithms.hh"

#include "../CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Return a simplified alias via its target.
 */
struct AliasSimplifier
{
    // Replace alias with target
    NodeId operator()(Aliased const& a) const { return a.node; }

    // Other types do not simplify further
    template<class T>
    NodeId operator()(T const&) const
    {
        return {};
    }
};

//---------------------------------------------------------------------------//
/*!
 * Return a simplified negation via its target.
 */
struct NegationSimplifier
{
    Node operator()(True const&) const { return False{}; }

    Node operator()(False const&) const { return True{}; }

    Node operator()(Aliased const& a) const
    {
        // Replace with target
        return Negated{a.node};
    }

    Node operator()(Negated const& n) const
    {
        // Replace a double-negative
        return Aliased{n.node};
    }

    // Other types do not simplify
    template<class T>
    Node operator()(T const&) const
    {
        return NodeSimplifier::no_simplification();
    }
};

//---------------------------------------------------------------------------//
//! Return a pointer to a node if joined in the given way, otherwise nullptr
Joined const* get_if_joined_like(Node const* n, OperatorToken op)
{
    auto const* j = std::get_if<Joined>(n);
    if (j && j->op != op)
    {
        // Same variant type, different join type
        j = nullptr;
    }
    return j;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with the tree to visit.
 */
NodeSimplifier::NodeSimplifier(CsgTree const& tree)
    : tree_{tree}, visit_node_{tree}
{
}

//---------------------------------------------------------------------------//
/*!
 * Replace aliased node.
 */
auto NodeSimplifier::operator()(Aliased const& a) const -> Node
{
    return Aliased{visit_node_(AliasSimplifier{}, a.node)};
}

//---------------------------------------------------------------------------//
/*!
 * Replace a negated node.
 */
auto NodeSimplifier::operator()(Negated const& n) const -> Node
{
    return visit_node_(NegationSimplifier{}, n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Simplify a joined node.
 *
 * This modifies the node in place to avoid copying the vector.
 */
auto NodeSimplifier::operator()(Joined& j) const -> Node
{
    CELER_EXPECT(j.op == op_and || j.op == op_or);

    // Return automatic replacement if this node is found
    auto const constant_node = (j.op == op_and ? CsgTree::false_node_id()
                                               : CsgTree::true_node_id());
    auto const ignore_node = (j.op == op_and ? CsgTree::true_node_id()
                                             : CsgTree::false_node_id());

    // Maintain a separate list of nodes to add to avoid invalidating iterators
    std::vector<NodeId> to_merge;

    // Replace any aliases in each daughter
    for (NodeId& d : j.nodes)
    {
        // Replace aliases first
        if (auto repl = visit_node_(AliasSimplifier{}, d))
        {
            d = repl;
        }

        if (d == constant_node)
        {
            // Short circuit logic based on the logical operator
            return Aliased{constant_node};
        }
        else if (d == ignore_node)
        {
            // Replace with a null ID (to be eliminated during sort/unique)
            d = NodeId{};
        }
        else if (Joined const* dj = get_if_joined_like(&tree_[d], j.op))
        {
            // Add to a merge queue to avoid invalidating iterators
            to_merge.insert(to_merge.end(), dj->nodes.begin(), dj->nodes.end());
            d = NodeId{};
        }
    }

    // Add any nodes to be merged into the expression after looping
    j.nodes.insert(j.nodes.end(), to_merge.begin(), to_merge.end());

    // Sort and uniquify the node IDs
    std::sort(j.nodes.begin(), j.nodes.end());
    j.nodes.erase(std::unique(j.nodes.begin(), j.nodes.end()), j.nodes.end());

    // Pop any ignored node, which will be a single one at the back (but the
    // updated list may be empty so check that first!)
    if (!j.nodes.empty() && !j.nodes.back())
    {
        j.nodes.pop_back();
    }
    // Double check that all nodes are valid (null nodes sort to the back)
    CELER_ASSERT(std::all_of(j.nodes.begin(), j.nodes.end(), Identity{}));

    if (j.nodes.empty())
    {
        // "all of" with no elements, or "any of" with no elements
        return Aliased{ignore_node};
    }

    if (j.nodes.size() == 1)
    {
        // Single-element join is just an alias
        return Aliased{j.nodes.front()};
    }

    // After merging daughter nodes and uniquifying, track the "negated" IDs of
    // encountered nodes to eliminate join<op>(A, ~A, ...)
    std::unordered_set<NodeId> negated;
    for (NodeId const& d : j.nodes)
    {
        if (negated.count(d))
        {
            // This negation of this node exists in the join expression
            // A & ~A -> F
            // A | ~A -> T
            return Aliased{constant_node};
        }

        if (auto negated_id = tree_.find(Negated{d}))
        {
            // The negated node exists somewhere in the tree; keep track to
            // compare it against other daughters
            negated.insert(negated_id);
        }
    }

    return std::move(j);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
