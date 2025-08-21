//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTree.cc
//---------------------------------------------------------------------------//
#include "CsgTree.hh"

#include <algorithm>
#include <utility>
#include <variant>

#include "corecel/cont/Range.hh"

#include "detail/NodeSimplifier.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
//! Check user input for validity
struct IsUserNodeValid
{
    size_type max_id{};

    bool operator()(orangeinp::True const&) const { return true; }
    bool operator()(orangeinp::False const&) const { return true; }
    bool operator()(orangeinp::Aliased const&) const { return true; }
    bool operator()(orangeinp::Negated const& n) const
    {
        return (n.node < this->max_id);
    }
    bool operator()(orangeinp::Surface const& s) const
    {
        return static_cast<bool>(s.id);
    }
    bool operator()(orangeinp::Joined const& j) const
    {
        return ((j.op == orangeinp::op_and) || (j.op == orangeinp::op_or))
               && std::all_of(
                   j.nodes.begin(),
                   j.nodes.end(),
                   [this](orangeinp::NodeId n) { return n < this->max_id; });
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Insert true and 'negated true', and define equivalence operations.
 */
CsgTree::CsgTree()
    : nodes_{orangeinp::True{}, orangeinp::Negated{true_node_id()}}
    , ids_{{Node{std::in_place_type<orangeinp::True>}, true_node_id()},
           {Node{std::in_place_type<orangeinp::False>}, false_node_id()},
           {orangeinp::Negated{true_node_id()}, false_node_id()},
           {orangeinp::Negated{false_node_id()}, true_node_id()}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Add a node of type T and return the new ID.
 *
 * This performs a single level of simplification.
 */
auto CsgTree::insert(Node&& n) -> Insertion
{
    CELER_EXPECT(!n.valueless_by_exception()
                 && std::visit(IsUserNodeValid{this->size()}, n));

    // Normalize and simplify in place up to one level
    if (this->simplify(n))
    {
        if (auto* a = std::get_if<orangeinp::Aliased>(&n))
        {
            // Simplified to an aliased node
            return {a->node, false};
        }
    }

    auto [iter, inserted] = ids_.emplace(std::move(n), NodeId{});
    if (inserted)
    {
        // Save new node ID
        iter->second = id_cast<NodeId>(nodes_.size());
        // Add a copy of the new node
        nodes_.push_back(iter->first);
    }
    return {iter->second, inserted};
}

//---------------------------------------------------------------------------//
/*!
 * Find the node ID of the CSG expression if it exists.
 *
 * This consumes the input expression in order to simplify it.
 */
NodeId CsgTree::find(Node&& n) const
{
    Node temp{std::move(n)};
    this->simplify(temp);
    if (auto* a = std::get_if<Aliased>(&temp))
    {
        // Node was simplified to an existing ID
        return a->node;
    }
    // Try the node as is
    auto iter = ids_.find(temp);
    if (iter == ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Replace a node with a simplified version or constant.
 *
 * Inserting the node performs one level of simplification.
 */
auto CsgTree::exchange(NodeId node_id, Node&& n) -> Node
{
    CELER_EXPECT(node_id > false_node_id());
    CELER_EXPECT(!n.valueless_by_exception()
                 && std::visit(IsUserNodeValid{node_id.unchecked_get()}, n));

    // Simplify the node first
    this->simplify(n);

    // Replace the node with an alias to a node deeper in the tree
    if (auto* a = std::get_if<orangeinp::Aliased>(&n))
    {
        CELER_ASSERT(a->node < node_id);
        return std::exchange(this->at(node_id), orangeinp::Aliased{a->node});
    }

    // Add the node to the map of deduplicated nodes
    auto [iter, inserted] = ids_.insert({std::move(n), node_id});
    if (inserted)
    {
        // Node representation doesn't exist elsewhere in the tree
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
    return std::exchange(this->at(node_id), orangeinp::Aliased{iter->second});
}

//---------------------------------------------------------------------------//
/*!
 * Perform a simplification of a node in-place.
 *
 * \return Simplified node
 */
auto CsgTree::simplify(NodeId node_id) -> Simplification
{
    auto repl = this->exchange(node_id, Node{this->at(node_id)});
    if (repl == this->at(node_id))
    {
        return {};
    }
    return Simplification{std::move(repl)};
}

//---------------------------------------------------------------------------//
/*!
 * Simplify a node expression.
 *
 * \return Whether simplification was performed
 */
bool CsgTree::simplify(Node& n) const
{
    Node repl = std::visit(detail::NodeSimplifier{*this}, n);
    if (repl != Node{detail::NodeSimplifier::no_simplification()})
    {
        n = std::move(repl);
        return true;
    }
    return false;
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
    for (auto n : range(orangeinp::NodeId(tree.size())))
    {
        os << n.unchecked_get() << ": " << tree[n] << ", ";
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
