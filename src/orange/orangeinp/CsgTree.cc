//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTree.cc
//---------------------------------------------------------------------------//
#include "CsgTree.hh"

#include <algorithm>
#include <tuple>
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

    {
        // Try to simplify the node up to one level when inserting.
        Node repl = std::visit(detail::NodeSimplifier{*this}, n);
        if (repl != Node{detail::NodeSimplifier::no_simplification()})
        {
            n = std::move(repl);
            if (auto* a = std::get_if<orangeinp::Aliased>(&n))
            {
                // Simplified to an aliased node
                return {a->node, false};
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
    return {iter->second, inserted};
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
    Node repl = std::visit(detail::NodeSimplifier{*this}, n);
    if (repl != Node{detail::NodeSimplifier::no_simplification()})
    {
        n = std::move(repl);
    }

    if (auto* a = std::get_if<orangeinp::Aliased>(&n))
    {
        return std::exchange(this->at(node_id), orangeinp::Aliased{a->node});
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
 * Simplify negated joins using De Morgan's law.
 *
 * This is required if the tree's logic expression is used with
 * \c InfixEvaluator as negated joins are not supported.
 */
void CsgTree::simplify_negated_joins()
{
    // Vector of node_id Negated{}
    std::vector<std::tuple<NodeId, Joined*>> stack;
    stack.reserve(nodes_.size());

    // First pass through all nodes to find all nand / nor
    for (auto const& [node, node_id] : ids_)
    {
        if (auto* negated = std::get_if<orangeinp::Negated>(&node))
        {
            if (auto* join
                = std::get_if<orangeinp::Joined>(&this->at(negated->node)))
            {
                // Current node is Negated{Joined{...}}
                stack.emplace_back(node_id, join);
            }
        }
    }

    while (!stack.empty())
    {
        // Get one node
        auto [negated_id, joined] = stack.back();
        stack.pop_back();

        Joined transformed;
        transformed.op = joined->op == op_and ? op_or : op_and;
        transformed.nodes.reserve(joined->nodes.size());

        // Negate all the join operands
        for (auto& join_operand : joined->nodes)
        {
            // Try to insert the negated operand
            auto [new_node, inserted] = this->insert(Negated{join_operand});
            if (inserted)
            {
                // Update the new join node with the negated operand
                transformed.nodes.push_back(new_node);
                if (auto* join
                    = std::get_if<orangeinp::Joined>(&this->at(join_operand)))
                {
                    // we just inserted a Negated{Join{...}}, need to simplify
                    stack.emplace_back(new_node, join);
                }
            }
        }
        // Override the old negated join node with the new join
        this->exchange(negated_id, std::move(transformed));
    }
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
