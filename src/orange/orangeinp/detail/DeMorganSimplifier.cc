//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/DeMorganSimplifier.cc
//---------------------------------------------------------------------------//

#include "DeMorganSimplifier.hh"

#include <utility>
#include <variant>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "orange/OrangeTypes.hh"

#include "../CsgTree.hh"
#include "../CsgTreeUtils.hh"
#include "../CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find the simplified node corresponding to the original node.
 *
 * For node_id in the original tree, find the equivalent node in the simplified
 * tree, i.e., either the DeMorgan simplification or the same node. Return a
 * null id if there are no equivalent.
 */
NodeId DeMorganSimplifier::MatchingNodes::equivalent_node() const
{
    if (simplified_to)
        return simplified_to;
    if (unmodified)
        return unmodified;
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Construct and fix bitsets size.
 */
DeMorganSimplifier::DeMorganSimplifier(CsgTree const& tree) : tree_(tree) {}

//---------------------------------------------------------------------------//
/*!
 * Access or create a translation entry.
 */
auto DeMorganSimplifier::translation(NodeId id) -> MatchingNodes&
{
    CELER_EXPECT(id < matching_nodes_.size());
    return matching_nodes_[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Perform the simplification. The state of the instance isn't cleared, so only
 * call this once.
 */
TransformedTree DeMorganSimplifier::operator()()
{
    // Mark nodes related to negated joins
    matching_nodes_.assign(tree_.size(), {});
    this->find_join_negations();

    // Save volume nodes
    is_volume_node_.assign(tree_.size(), false);
    for (auto node_id : tree_.volumes())
    {
        CELER_ASSERT(node_id < is_volume_node_.size());
        is_volume_node_[node_id.unchecked_get()] = true;
    }

    // Perform simplification
    CsgTree simplified_tree = this->build_simplified_tree();

    // Find equivalent nodes
    std::vector<NodeId> equivalent_nodes(tree_.size());
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        equivalent_nodes[node_id.get()]
            = this->translation(node_id).equivalent_node();
    }

    return {simplified_tree, equivalent_nodes};
}

//---------------------------------------------------------------------------//
/*!
 * Get a non-aliased Node variant
 */
Node const& DeMorganSimplifier::get_node(NodeId node_id) const
{
    CELER_EXPECT(node_id < tree_.size());
    NodeId dealiased{node_id};
    while (auto const* aliased = std::get_if<Aliased>(&tree_[dealiased]))
    {
        dealiased = aliased->node;
        CELER_ASSERT(dealiased < tree_.size());
    }
    return tree_[dealiased];
}

//---------------------------------------------------------------------------//
/*!
 * First pass through the tree to find negated set operations and parents of
 * each node.
 */
void DeMorganSimplifier::find_join_negations()
{
    ScopedProfiling profile_this{"orange-demorgan-find"};
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        auto const& node = this->get_node(node_id);
        if (auto* negated = std::get_if<Negated>(&node))
        {
            // Mark the parent relationship
            this->insert_parent(node_id, negated->node);

            if (std::holds_alternative<Joined>(this->get_node(negated->node)))
            {
                this->insert_negated_children(negated->node);
            }
        }
        else if (auto* joined = std::get_if<Joined>(&node))
        {
            for (auto const& child_node_id : joined->nodes)
            {
                this->insert_parent(node_id, child_node_id);
            }
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Mark that the first node is a parent of the second.
 *
 * \param node_id a \c Joined node_id with a \c Negated parent.
 */
bool DeMorganSimplifier::insert_parent(NodeId parent, NodeId child)
{
    CELER_EXPECT(parent);
    CELER_EXPECT(child);
    CELER_EXPECT(parent >= child);

    // Insert child row
    auto&& [iter, inserted] = parents_.try_emplace(child);
    CELER_ASSERT(iter != parents_.end());
    // Insert parent entry
    inserted = iter->second.insert(parent).second;

    return inserted;
}

//---------------------------------------------------------------------------//
/*!
 * Recursively record that we need to insert \c Negated node for operands of a
 * \c Joined node.
 *
 * \param node_id a \c Joined node_id with a \c Negated parent.
 */
bool DeMorganSimplifier::insert_negated_children(NodeId node_id)
{
    CELER_ASSUME(std::holds_alternative<Joined>(this->get_node(node_id)));
    auto&& [iter, inserted] = negated_join_nodes_.insert(node_id);
    if (!inserted)
    {
        return inserted;
    }
    auto& [op, operands] = std::get<Joined>(tree_[node_id]);

    for (auto const& child_node : operands)
    {
        Node const& target_node = this->get_node(child_node);
        if (std::holds_alternative<Joined>(target_node))
        {
            // This negated join node has a join operand, so we'll have to
            // insert a negated join of that join operand and its operands
            this->insert_negated_children(child_node);
        }
        else if (!std::holds_alternative<Negated>(target_node))
        {
            // Negate each operand unless it's a negated node, in which
            // case double negation will cancel to the child of that operand
            new_negated_nodes_.insert(child_node);
        }
    }
    return inserted;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the second node is a child of the first.
 */
bool DeMorganSimplifier::is_parent_of(NodeId parent, NodeId child) const
{
    auto iter = parents_.find(child);
    if (iter == parents_.end())
    {
        return false;
    }
    return iter->second.count(parent);
}

//---------------------------------------------------------------------------//
/*!
 * Second pass through the tree to build the simplified tree.
 *
 * \return the simplified tree.
 */
CsgTree DeMorganSimplifier::build_simplified_tree()
{
    ScopedProfiling profile_this{"orange-demorgan-simplify"};
    CsgTree result{};

    // We can now build the new tree
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        // Special handling for negated and joined nodes
        if (!this->process_negated_joined_nodes(node_id, result))
        {
            continue;
        }

        // This node needs to be inserted in the simplified tree, but we need
        // to update the node ids of its children

        // deref aliased nodes, we don't want to insert them in the new tree
        Node new_node = this->get_node(node_id);

        CELER_ASSERT(!std::holds_alternative<Aliased>(new_node));

        if (auto* negated = std::get_if<Negated>(&new_node))
        {
            // We're never inserting a negated node pointing to a
            // joined or negated node so it's child must have an
            // unmodified equivalent in the simplified tree
            auto& trans = this->translation(negated->node);
            CELER_ASSERT(trans.unmodified);
            negated->node = trans.unmodified;
        }
        else if (auto* joined = std::get_if<Joined>(&new_node))
        {
            // This is not a negated join, they are inserted in
            // process_negated_joined_nodes
            for (auto& child : joined->nodes)
            {
                // That means we should find an equivalent node for
                // each operand, either a simplified negated join or an
                // unmodified node
                auto& trans = this->translation(child);
                CELER_ASSERT(trans.equivalent_node());
                child = trans.equivalent_node();
            }
        }

        auto [new_id, inserted] = result.insert(std::move(new_node));
        auto& trans = this->translation(node_id);

        CELER_ASSERT(!trans.unmodified);
        // Record the new node id for parents of that node
        trans.unmodified = new_id;

        // We might have to insert a negated version of that node
        if (new_negated_nodes_.count(node_id))
        {
            Node const& target_node{this->get_node(node_id)};
            CELER_ASSERT(!std::holds_alternative<Negated>(target_node)
                         && !std::holds_alternative<Joined>(target_node)
                         && !trans.new_negation);
            auto [new_negated_node_id, negated_inserted]
                = result.insert(Negated{new_id});
            trans.new_negation = new_negated_node_id;
        }
    }

    // Set the volumes in the simplified tree by checking the translation map
    for (auto volume : tree_.volumes())
    {
        // Volumes should be kept, so we must have an equivalent node in the
        // new tree.
        // This is not always the exact same node, e.g., if the volume
        // points to a negated join, it will still be simplified
        auto& trans = this->translation(volume);
        CELER_ASSERT(trans.equivalent_node());
        result.insert_volume(trans.equivalent_node());
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Special handling for a \c Joined or \c Negated node. A Joined node can be
 * duplicated if it has negated and non-negated parents.
 * Similarly, a Negated node might have to be omitted because its only parents
 * are negated nodes.
 *
 * Determine whether the negated or joined node should be inserted in the
 * simplified tree. In addition, if the joined has negated parents and must be
 * inserted in the simplified tree, do the insertion.
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
    Node const& target_node = this->get_node(node_id);
    if (auto const* negated = std::get_if<Negated>(&target_node))
    {
        // This node has a joined child, we must never insert it in the
        // simplified tree
        if (std::holds_alternative<Joined>(this->get_node(negated->node)))
        {
            // Redirect parents looking for this node to the new Joined
            // node which is logically equivalent
            auto& trans = this->translation(negated->node);
            CELER_ASSERT(trans.opposite_join);
            this->translation(node_id).simplified_to = trans.opposite_join;
            return false;
        }

        // From here we know this isn't the negation of a join
        // operation

        // Check if the negation is a root or a volume. If so, we must
        // insert it in the simplified tree
        if (is_volume_node_[node_id.get()])
        {
            return true;
        }
        auto parents_iter = parents_.find(node_id);
        if (parents_iter == parents_.end())
        {
            // No parents: root node
            return true;
        }

        // Loop over all parents of node_id
        for (NodeId p : parents_iter->second)
        {
            // A negated node should never have a negated parent
            CELER_ASSERT(!std::holds_alternative<Negated>(this->get_node(p)));

            // If an ancestor is a join node that should be inserted
            // unmodified, this negated node is still necessary
            if (std::holds_alternative<Joined>(this->get_node(p))
                && this->should_insert_join(p))
            {
                return true;
            }
        }

        // Otherwise, we only have negated joins as ancestor, so this
        // is no longer necessary in the simplified tree
        return false;
    }
    else if (auto const* joined = std::get_if<Joined>(&target_node))
    {
        // Check if this node needs a simplification
        if (negated_join_nodes_.count(node_id))
        {
            // Insert the negated node
            auto [new_id, inserted]
                = result.insert(this->build_negated_node(*joined));
            // Record that we inserted an opposite join for that node
            this->translation(node_id).opposite_join = std::move(new_id);
        }
        return this->should_insert_join(node_id);
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Create an opposite \c Joined node.
 *
 * \param joined the \c Joined node to process.
 *
 * \return Join node with opposite operation and negated operands
 */
Joined DeMorganSimplifier::build_negated_node(Joined const& joined)
{
    // Insert the opposite join
    auto const& [op, nodes] = joined;
    std::vector<NodeId> operands;
    operands.reserve(nodes.size());

    // Negate each operand, pointing to node ids in the simplified
    // tree
    for (auto n : nodes)
    {
        // Negation of a negated operand cancel each other, we can
        // just use the child of that negated operand
        if (auto const* neg = std::get_if<Negated>(&this->get_node(n)))
        {
            // We should have recorded that this node was necessary
            // for a join
            auto& trans = this->translation(neg->node);
            CELER_ASSERT(trans.unmodified);
            operands.push_back(trans.unmodified);
        }
        else
        {
            // Otherwise, we should have inserted a negated
            // version of that operand in the simplified tree.
            // It's either a simplified join or a negated node
            operands.push_back([&] {
                auto& trans = this->translation(n);
                CELER_ASSERT(trans.new_negation || trans.opposite_join);
                if (trans.new_negation)
                    return trans.new_negation;
                return trans.opposite_join;
            }());
        }
    }
    return Joined{(op == logic::land) ? logic::lor : logic::land,
                  std::move(operands)};
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
    CELER_EXPECT(std::holds_alternative<Joined>(this->get_node(node_id)));

    // TODO: logic is very similar to process_negated_joined_nodes

    // This join node is referred by a volume or a root node, we must insert it
    if (is_volume_node_[node_id.get()])
    {
        return true;
    }

    auto parents_iter = parents_.find(node_id);
    if (parents_iter == parents_.end())
    {
        // No parents: root node
        return true;
    }

    // We must insert the original join node if one of the following is true
    // 1. It has a Join ancestor that is not negated
    // 2. It has a negated parent, and that negated node has a negated join
    // parent (double negation of that join)
    // Loop over all parents of node_id
    for (NodeId p : parents_iter->second)
    {
        // Check if a parent requires that node to be inserted
        // TODO: Is it really correct in all cases...
        Node const& dealiased = this->get_node(p);

        if (std::holds_alternative<Joined>(dealiased)
            && this->should_insert_join(p))
        {
            return true;
        }
        if (std::holds_alternative<Negated>(dealiased))
        {
            // Loop over grandparents
            auto gp_iter = parents_.find(p);
            if (gp_iter != parents_.end())
            {
                for (NodeId gp : gp_iter->second)
                {
                    if (this->is_parent_of(gp, p)
                        && negated_join_nodes_.count(gp))
                    {
                        // Has negated join
                        return true;
                    }
                }
            }
        }
    }

    // If not, we don't insert it
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
