//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/CsgTree.cc
//---------------------------------------------------------------------------//
#include "CsgTree.hh"

#include <algorithm>
#include <tuple>
#include <utility>
#include <variant>

#include "corecel/cont/Range.hh"

#include "detail/NodeReplacementInserter.hh"
#include "detail/NodeSimplifier.hh"

using namespace celeritas::csg;

#include <iostream>
using std::cout;
using std::endl;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Check user input for validity
struct IsUserNodeValid
{
    size_type max_id{};

    bool operator()(csg::True const&) const { return true; }
    bool operator()(csg::False const&) const { return true; }
    bool operator()(csg::Aliased const&) const { return true; }
    bool operator()(csg::Negated const& n) const
    {
        return (n.node < this->max_id);
    }
    bool operator()(csg::Surface const& s) const
    {
        return static_cast<bool>(s.id);
    }
    bool operator()(csg::Joined const& j) const
    {
        return ((j.op == csg::op_and) || (j.op == csg::op_or))
               && std::all_of(
                   j.nodes.begin(), j.nodes.end(), [this](csg::NodeId n) {
                       return n < this->max_id;
                   });
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Insert true and false and 'negated true' (which redirects to false).
 */
CsgTree::CsgTree()
    : nodes_{csg::True{}, csg::False{}}
    , ids_{{Node{std::in_place_type<csg::True>}, true_node_id()},
           {Node{std::in_place_type<csg::False>}, false_node_id()},
           {csg::Negated{true_node_id()}, false_node_id()}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Add a node of type T and return the new ID.
 *
 * This performs a single level of simplification on "joined" nodes only, since
 * the user shouldn't be inserting aliased nodes.
 */
auto CsgTree::insert(Node&& n) -> NodeId
{
    CELER_EXPECT(!n.valueless_by_exception()
                 && std::visit(IsUserNodeValid{this->size()}, n));

    if (auto* j = std::get_if<csg::Joined>(&n))
    {
        // Replace "joined" with a simplified node
        Node repl = NodeSimplifier{*this}(*j);
        CELER_ASSERT(repl != Node{NodeSimplifier::no_simplification()});
        if (auto* a = std::get_if<csg::Aliased>(&repl))
        {
            // Joined simplified to an aliased node
            return a->node;
        }
        n = std::move(repl);
    }

    auto [iter, inserted] = ids_.insert({std::move(n), {}});
    if (inserted)
    {
        // Save new node ID
        iter->second = NodeId{static_cast<size_type>(nodes_.size())};
        // Add a copy of the new node
        nodes_.push_back(iter->first);
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Replace a node with a simplified version or constant.
 */
auto CsgTree::exchange(NodeId node_id, Node&& n) -> Node
{
    CELER_EXPECT(node_id > false_node_id());
    CELER_EXPECT(!n.valueless_by_exception()
                 && std::visit(IsUserNodeValid{node_id.unchecked_get()}, n));

    //! Simplify the node first
    Node repl = std::visit(NodeSimplifier{*this}, n);
    if (repl != Node{NodeSimplifier::no_simplification()})
    {
        cout << "\n  ... simplified to replacement: " << repl;
        n = std::move(repl);
    }

    if (auto* a = std::get_if<csg::Aliased>(&n))
    {
        cout << "\n  ... returning alias" << n;
        return std::exchange(this->at(node_id), csg::Aliased{a->node});
    }

    // Add the node to the map of deduplicated nodes
    auto [iter, inserted] = ids_.insert({std::move(n), NodeId{}});
    if (inserted)
    {
        // Node representation doesn't exist elsewhere in the tree
        iter->second = node_id;
        cout << "\n  ... new node";
        return std::exchange(this->at(node_id), Node{iter->first});
    }
    if (iter->second == node_id)
    {
        // No change
        cout << "\n  ... no change";
        return (*this)[node_id];
    }
    if (iter->second > node_id)
    {
        using std::swap;
        // A node *higher* in the tree is equivalent to this one: swap the
        // definitions so that the higher aliases the lower
        cout << "\n  ... swapping IDs";
        swap(this->at(iter->second), this->at(node_id));
        swap(iter->second, node_id);
    }

    // Replace the more complex definition with an alias to a lower ID
    CELER_ASSERT(iter->second < node_id);
    cout << "\n  ... aliasing " << node_id.get() << " to "
         << iter->second.get();
    return std::exchange(this->at(node_id), csg::Aliased{iter->second});
}

//---------------------------------------------------------------------------//
/*!
 * Perform a simplification of a node in-place.
 *
 * \return Whether simplification took place
 */
bool CsgTree::simplify(NodeId node_id)
{
    auto repl = this->exchange(node_id, Node{this->at(node_id)});
    cout << "\n ... comparing exchanged " << repl << " to current "
         << this->at(node_id);
    return repl != this->at(node_id);
}

//---------------------------------------------------------------------------//
/*!
 * Get a mutable node.
 */
auto CsgTree::at(NodeId node_id) -> Node&
{
    CELER_EXPECT(node_id < nodes_.size());
    return nodes_[node_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Print the tree's contents.
 */
std::ostream& operator<<(std::ostream& os, CsgTree const& tree)
{
    os << "{\n";
    for (auto n : range(csg::NodeId(tree.size())))
    {
        os << n.unchecked_get() << ": " << tree[n] << ",\n";
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Replace the given node ID with the replacement node.
 *
 * \return Node ID of the lowest node that required simplification.
 */
NodeId replace_down(CsgTree* tree, NodeId n, Node repl)
{
    CELER_EXPECT(is_boolean_node(repl));

    // Recursively apply implications of setting to boolean `b`
    // - "negate": non-constant daughter node is replaced with ~b
    // - "replace": non-constant daughter node is replaced with `b`
    // - "join": for (false, or): all daughters are "false"
    //           for (true, and): all daughters are "true"
    // - surface: "true"
    // - constant: check for contradiction
    NodeReplacementInserter::VecNode stack{{n, std::move(repl)}};

    NodeId lowest_node{n};

    while (!stack.empty())
    {
        n = std::move(stack.back().first);
        repl = std::move(stack.back().second);
        stack.pop_back();
        lowest_node = std::min(n, lowest_node);

        cout << "Replacing " << n.get() << " with " << repl << ":";
        Node prev = tree->exchange(n, std::move(repl));
        cout << "\n  ==> " << prev << endl;
        std::visit(NodeReplacementInserter{&stack, repl}, prev);
        cout << "Stack size: " << stack.size() << endl;
    }
    return lowest_node;
}

//---------------------------------------------------------------------------//
/*!
 * Simplify all nodes in the tree starting with this one.
 *
 * Running this successfuly starting with the lowest node to fully simplify the
 * tree. The number of passes is *at most* equal to the depth of the tree.
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
        cout << "Simpifying " << node_id.get() << ": " << (*tree)[node_id];
        bool simplified = tree->simplify(node_id);
        cout << "\n  ==> ";
        if (simplified)
        {
            cout << (*tree)[node_id];
        }
        else
        {
            cout << " (no change)";
        }
        cout << endl;
        if (simplified && !result)
        {
            result = node_id;
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
