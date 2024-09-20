//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "corecel/cont/VariantUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create the matrix with the given extent size.
 */
DeMorganSimplifier::Matrix2D::Matrix2D(size_type extent) noexcept
    : extent_(extent)
{
    data_.resize(extent_ * extent_);
}

//---------------------------------------------------------------------------//
/*!
 * Access the element at the given index.
 */
std::vector<bool>::reference
DeMorganSimplifier::Matrix2D::operator[](indices index)
{
    auto& [row, col] = index;
    CELER_EXPECT(row < extent_ && col < extent_);
    return data_[row.get() * extent_ + col.get()];
}

//---------------------------------------------------------------------------//
/*!
 * The extent along one dimension.
 */
size_type DeMorganSimplifier::Matrix2D::extent() const noexcept
{
    return extent_;
}

//---------------------------------------------------------------------------//
/*!
 * For node_id in the original tree, find the equivalent node in the simplified
 * tree, i.e., either the DeMorgan simplification or the same node, return an
 * invalid id if there are no equivalent.
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
DeMorganSimplifier::DeMorganSimplifier(CsgTree const& tree)
    : tree_(tree), parents_(tree_.size())
{
    new_negated_nodes_.resize(tree_.size());
    negated_join_nodes_.resize(tree_.size());
    node_ids_translation_.resize(tree_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Perform the simplification. The state of the instance isn't cleared, so only
 * call this once.
 */
CsgTree DeMorganSimplifier::operator()()
{
    this->find_join_negations();
    return this->build_simplified_tree();
}

//---------------------------------------------------------------------------//
/*!
 * Find the real node id to access by dereferencing Aliased nodes.
 */
NodeId DeMorganSimplifier::dealias(NodeId node_id) const
{
    NodeId dealiased{node_id};
    while (auto const* aliased = std::get_if<Aliased>(&tree_[dealiased]))
    {
        dealiased = aliased->node;
    }
    return dealiased;
}

//---------------------------------------------------------------------------//
/*!
 * First pass through the tree to find negated set operations and parents of
 * each node.
 */
void DeMorganSimplifier::find_join_negations()
{
    for (auto node_id : range(NodeId{tree_.size()}))
    {
        auto const* node = &tree_[this->dealias(node_id)];
        if (auto* negated = std::get_if<Negated>(node))
        {
            parents_[{negated->node, node_id}] = true;
            parents_[{negated->node, has_parents_index_}] = true;
            if (std::holds_alternative<Joined>(
                    tree_[this->dealias(negated->node)]))
            {
                // This is a negated join node
                negated_join_nodes_[negated->node.get()] = true;
                this->add_negation_for_operands(negated->node);
            }
        }
        else if (auto* joined = std::get_if<Joined>(node))
        {
            for (auto const& join_operand : joined->nodes)
            {
                parents_[{join_operand, node_id}] = true;
                parents_[{join_operand, has_parents_index_}] = true;
            }
        }
    }

    // Volumes act as parents of nodes.
    // We can reuse node id 0 to set that a node has a parent volume
    for (auto node_id : tree_.volumes())
    {
        parents_[{node_id, is_volume_index_}] = true;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Recursively record that we need to insert \c Negated node for operands of a
 * \c Joined node.
 *
 * \param node_id a \c Joined node_id with a \c Negated parent.
 */
void DeMorganSimplifier::add_negation_for_operands(NodeId node_id)
{
    CELER_ASSUME(std::holds_alternative<Joined>(tree_[this->dealias(node_id)]));
    auto& [op, operands] = std::get<Joined>(tree_[node_id]);

    for (auto const& join_operand : operands)
    {
        Node const& target_node = tree_[this->dealias(join_operand)];
        if (std::holds_alternative<Joined>(target_node))
        {
            // This negated join node has a join operand, so we'll have to
            // insert a negated join of that join operand and its operands
            negated_join_nodes_[join_operand.get()] = true;
            this->add_negation_for_operands(join_operand);
        }
        else if (!std::holds_alternative<Negated>(target_node))
        {
            // Negate each operand unless it's a negated node, in which
            // case double negation will cancel to the child of that operand
            new_negated_nodes_[join_operand.get()] = true;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Second pass through the tree to build the simplified tree.
 *
 * \return the simplified tree.
 */
CsgTree DeMorganSimplifier::build_simplified_tree()
{
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
        Node new_node = tree_[this->dealias(node_id)];

        CELER_ASSERT(!std::holds_alternative<Aliased>(new_node));

        if (auto* negated = std::get_if<Negated>(&new_node))
        {
            // We're never inserting a negated node pointing to a
            // joined or negated node so it's child must have an
            // unmodified equivalent in the simplified tree
            CELER_EXPECT(node_ids_translation_[negated->node.get()].unmodified);
            negated->node
                = node_ids_translation_[negated->node.get()].unmodified;
        }
        else if (auto* joined = std::get_if<Joined>(&new_node))
        {
            // This is not a negated join, they are inserted in
            // process_negated_joined_nodes
            for (auto& op : joined->nodes)
            {
                // That means we should find an equivalent node for
                // each operand, either a simplified negated join or an
                // unmodified node
                CELER_EXPECT(node_ids_translation_[op.get()].equivalent_node());
                op = node_ids_translation_[op.get()].equivalent_node();
            }
        }

        auto [new_id, inserted] = result.insert(std::move(new_node));
        auto& trans = node_ids_translation_[node_id.get()];

        CELER_EXPECT(!trans.unmodified);
        // Record the new node id for parents of that node
        trans.unmodified = new_id;

        // We might have to insert a negated version of that node
        if (new_negated_nodes_[node_id.get()])
        {
            Node const& target_node{tree_[this->dealias(node_id)]};
            CELER_EXPECT(!std::holds_alternative<Negated>(target_node)
                         && !std::holds_alternative<Joined>(target_node)
                         && !trans.new_negation);
            auto [new_negated_node_id, negated_inserted]
                = result.insert(Negated{new_id});
            trans.new_negation = new_negated_node_id;
        }
    }

    // set the volumes in the simplified tree by checking the translation map
    for (auto volume : tree_.volumes())
    {
        // Volumes should be kept, so we must have an equivalent node in the
        // new tree.
        // This is not always the exact same node, e.g., if the volume
        // points to a negated join, it will still be simplified
        CELER_EXPECT(node_ids_translation_[volume.get()].equivalent_node());
        result.insert_volume(
            node_ids_translation_[volume.get()].equivalent_node());
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
    Node const* target_node = &tree_[this->dealias(node_id)];
    if (auto const* negated = std::get_if<Negated>(target_node))
    {
        // This node has a joined child, we must never insert it in the
        // simplified tree
        if (std::holds_alternative<Joined>(tree_[this->dealias(negated->node)]))
        {
            // Redirect parents looking for this node to the new Joined
            // node which is logically equivalent
            CELER_EXPECT(
                node_ids_translation_[negated->node.get()].opposite_join);
            node_ids_translation_[node_id.get()].simplified_to
                = node_ids_translation_[negated->node.get()].opposite_join;
            return false;
        }

        // From here we know this isn't the negation of a join
        // operation

        // Check if the negation is a root or a volume. If so, we must
        // insert it in the simplified tree
        if (parents_[{node_id, is_volume_index_}]
            || !parents_[{node_id, has_parents_index_}])
        {
            return true;
        }

        for (auto p : range(first_node_id_, NodeId{tree_.size()}))
        {
            // Not a parent
            if (!parents_[{node_id, p}])
                continue;

            // A negated node should never have a negated parent
            CELER_EXPECT(
                !std::holds_alternative<Negated>(tree_[this->dealias(p)]));

            // If an ancestor is a join node that should be inserted
            // unmodified, this negated node is still necessary
            if (std::holds_alternative<Joined>(tree_[this->dealias(p)])
                && this->should_insert_join(p))
            {
                return true;
            }
        }

        // Otherwise, we only have negated joins as ancestor, so this
        // is no longer necessary in the simplified tree
        return false;
    }
    else if (auto const* joined = std::get_if<Joined>(target_node))
    {
        // Check if this node needs a simplification
        if (negated_join_nodes_[node_id.get()])
        {
            // Insert the negated node
            auto [new_id, inserted]
                = result.insert(this->build_negated_node(*joined));
            // Record that we inserted an opposite join for that node
            node_ids_translation_[node_id.get()].opposite_join
                = std::move(new_id);
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
Joined DeMorganSimplifier::build_negated_node(Joined const& joined) const
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
        if (auto const* neg = std::get_if<Negated>(&tree_[this->dealias(n)]))
        {
            // We should have recorded that this node was necessary
            // for a join
            CELER_EXPECT(node_ids_translation_[neg->node.get()].unmodified);
            operands.push_back(
                node_ids_translation_[neg->node.get()].unmodified);
        }
        else
        {
            // Otherwise, we should have inserted a negated
            // version of that operand in the simplified tree.
            // It's either a simplified join or a negated node
            operands.push_back([&] {
                auto& trans = node_ids_translation_[n.get()];
                CELER_EXPECT(trans.new_negation || trans.opposite_join);
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
    CELER_EXPECT(std::holds_alternative<Joined>(tree_[this->dealias(node_id)]));

    // This join node is referred by a volume or a root node, we must insert it
    if (parents_[{node_id, is_volume_index_}]
        || !parents_[{node_id, has_parents_index_}])
    {
        return true;
    }

    // We must insert the original join node if one of the following is true
    // 1. It has a Join ancestor that is not negated
    // 2. It has a negated parent, and that negated node has a negated join
    // parent (double negation of that join)
    auto has_negated_join_parent = [&](NodeId n) {
        for (auto p : range(first_node_id_, NodeId{tree_.size()}))
        {
            if (parents_[{n, p}] && negated_join_nodes_[p.get()])
                return true;
        }
        return false;
    };

    for (auto p : range(first_node_id_, NodeId{tree_.size()}))
    {
        // Not a parent
        if (!parents_[{node_id, p}])
            continue;

        // Check if a parent requires that node to be inserted
        // TODO: Is it really correct in all cases...
        if (Node const& dealiased{tree_[this->dealias(p)]};
            (std::holds_alternative<Joined>(dealiased)
             && this->should_insert_join(p))
            || (std::holds_alternative<Negated>(dealiased)
                && has_negated_join_parent(p)))
        {
            return true;
        }
    }

    // If not, we don't insert it
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
