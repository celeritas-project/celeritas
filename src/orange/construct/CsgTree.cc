//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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

#include "detail/NodeSimplifier.hh"

using namespace celeritas::csg;

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
 * Insert true and 'negated true', and define equivalence operations.
 */
CsgTree::CsgTree()
    : nodes_{csg::True{}, csg::Negated{true_node_id()}}
    , ids_{{Node{std::in_place_type<csg::True>}, true_node_id()},
           {Node{std::in_place_type<csg::False>}, false_node_id()},
           {csg::Negated{true_node_id()}, false_node_id()},
           {csg::Negated{false_node_id()}, true_node_id()}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Add a node of type T and return the new ID.
 *
 * This performs a single level of simplification.
 */
auto CsgTree::insert(Node&& n) -> NodeId
{
    CELER_EXPECT(!n.valueless_by_exception()
                 && std::visit(IsUserNodeValid{this->size()}, n));

    {
        // Try to simplify the node up to one level when inserting.
        Node repl = std::visit(NodeSimplifier{*this}, n);
        if (repl != Node{NodeSimplifier::no_simplification()})
        {
            n = std::move(repl);
            if (auto* a = std::get_if<csg::Aliased>(&n))
            {
                // Simplified to an aliased node
                return a->node;
            }
        }
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
        n = std::move(repl);
    }

    if (auto* a = std::get_if<csg::Aliased>(&n))
    {
        return std::exchange(this->at(node_id), csg::Aliased{a->node});
    }

    // Add the node to the map of deduplicated nodes
    auto [iter, inserted] = ids_.insert({std::move(n), NodeId{}});
    if (inserted)
    {
        // Node representation doesn't exist elsewhere in the tree
        iter->second = node_id;
        return std::exchange(this->at(node_id), Node{iter->first});
    }
    if (iter->second == node_id)
    {
        // No change
        return (*this)[node_id];
    }
    if (iter->second > node_id)
    {
        using std::swap;
        // A node *higher* in the tree is equivalent to this one: swap the
        // definitions so that the higher aliases the lower
        swap(this->at(iter->second), this->at(node_id));
        swap(iter->second, node_id);
    }

    // Replace the more complex definition with an alias to a lower ID
    CELER_ASSERT(iter->second < node_id);
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
    os << '{';
    for (auto n : range(csg::NodeId(tree.size())))
    {
        os << n.unchecked_get() << ": " << tree[n] << ", ";
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
