//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/NodeSimplifier.cc
//---------------------------------------------------------------------------//
#include "NodeSimplifier.hh"

#include <algorithm>
#include <utility>

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
/*!
 * Return a simplified negation via its target.
 */
struct IsJoinedLike
{
    OperatorToken op;

    Joined const* operator()(Joined const& j) const
    {
        return j.op == op ? &j : nullptr;
    }

    // Other types are not joinable
    template<class T>
    Joined const* operator()(T const&) const
    {
        return nullptr;
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with the tree to visit.
 */
NodeSimplifier::NodeSimplifier(CsgTree const& tree) : visit_node_{tree} {}

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

    std::vector<NodeId> to_merge;

    // Replace any aliases in each daughter
    for (NodeId& d : j.nodes)
    {
        // Replace first
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
            // Replace with a null ID that will be sorted to the back of the
            // list
            d = NodeId{};
        }
        else if (Joined const* dj = visit_node_(IsJoinedLike{j.op}, d))
        {
            // Add nodes to merge; use a new list to avoid invalidating
            // iterators
            to_merge.insert(to_merge.end(), dj->nodes.begin(), dj->nodes.end());
            d = NodeId{};
        }
    }

    // TODO: we can look for combinations of A + ~A, returning "false" for
    // op_and or "true" for op_or

    // Add any merged nodes
    j.nodes.insert(j.nodes.end(), to_merge.begin(), to_merge.end());

    // Sort and uniquify the node ID
    std::sort(j.nodes.begin(), j.nodes.end());
    j.nodes.erase(std::unique(j.nodes.begin(), j.nodes.end()), j.nodes.end());

    // Pop any ignored node, which will be a single one at the back (but the
    // given list may be empty so check that first!)
    if (!j.nodes.empty() && !j.nodes.back())
    {
        j.nodes.pop_back();
    }

    if (j.nodes.empty())
    {
        // "all of" with no elements, or "any of" with no elements
        return Aliased{ignore_node};
    }

    if (j.nodes.size() == 1)
    {
        return Aliased{j.nodes.front()};
    }

    /*!
     * \todo implement De Morgan's laws to reduce the number of negations
     * - if all daughters are 'not', replace with nand/nor, and add support
     *   to logic stack
     * - OR add simplification strategy to csg tree, which may be tricky
     *   because that operation could modify the tree in place as
     *   well as increase the node depth
     */

    return std::move(j);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
