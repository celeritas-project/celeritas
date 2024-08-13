//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/DeMorganSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once
#include <unordered_map>
#include <unordered_set>

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
 * Implement DeMorgan transformations on a \c CsgTree.
 *
 * This class serves as an helper for \c
 * celeritas::orangeinp::transform_negated_joins implementation. It applied
 * DeMorgan's law on a \c CsgTree so that all negations of a set operator are
 * removed and transformed into their equivalent operation.
 *
 * Instances of this class shouldn't outlive the \c CsgTree used to construct
 * it as we keep a reference to it. Do not modify the original tree during the
 * simplification.
 */
class DeMorganSimplifier
{
  public:
    //! Construct a simplifier for the given tree
    explicit DeMorganSimplifier(CsgTree const& tree) : tree_(tree) {}

    // Simplify
    CsgTree operator()();

  private:
    //! Helper struct to translate ids from the original tree to ids in the
    //! simplified tree
    struct MatchingNodes
    {
        //! Set if a node as the exact same node in the simplified tree
        std::optional<NodeId> unmodified;

        //! Set if a node redirect to a different node, e.g. A Negated node
        //! pointing to a Join now redirects to the opposite join, or a double
        //! Negated node redirects to the non-negated child
        std::optional<NodeId> modified;

        //! Whether any node id is set
        explicit operator bool() const noexcept
        {
            return modified || unmodified;
        }

        // Check unmodified, then modified or default
        NodeId unmod_mod_or(NodeId default_id) const noexcept;

        // Check modified, then unmodified or default
        NodeId mod_unmod_or(NodeId default_id) const noexcept;
    };

    //!@{
    //! \name Set of NodeId
    using NodeIdSet = std::unordered_set<NodeId>;
    //! \name Map of NodeId, caching a pointer to the concrete type of a Node
    //! in the tree
    template<class T>
    using CachedNodeMap = std::unordered_map<NodeId, T const*>;
    //!@}

    // Unmark and node and its descendents
    void remove_orphaned(NodeId);

    // Signal that this node is a negated operation and that it should be
    // simplified
    void mark_negated_operator(NodeId);

    // Add negated nodes for operands of a Joined node
    void add_new_negated_nodes(NodeId);

    // Special handling for a Joined or Negated node
    bool process_negated_joined_nodes(NodeId, CsgTree&);

    // First pass through the tree to find negated set operations
    void find_negated_joins();

    // Second pass through the tree to build the simplified tree
    CsgTree build_simplified_tree();

    // Handle a negated Joined node
    CachedNodeMap<Joined>::mapped_type process_negated_node(NodeId);

    // Handle an orphaned node
    NodeIdSet::value_type process_orphaned_node(NodeId);

    // Check if a new Negated node must be inserted
    NodeIdSet::value_type process_new_negated_node(NodeId);

    // Handle a Negated node with a Joined child
    CachedNodeMap<Negated>::mapped_type process_transformed_negate_node(NodeId);

    //! the tree to simplify
    CsgTree const& tree_;

    //! For each node_id in that set, we'll create a Negated{node_id} node in
    //! the new tree.
    NodeIdSet new_negated_nodes_;

    //! We need to insert a negated version of these Joined nodes.
    //! The value is the Node referred to by the key as to avoid an extra
    //! get_if.
    CachedNodeMap<Joined> negated_join_nodes_;

    //! Contains the node_id of Negated{Joined}. These don't need to be
    //! inserted in the new tree and its parent can directly point to the
    //! transformed Joined node.
    CachedNodeMap<Negated> transformed_negated_nodes_;

    //! An orphan node should not be present in the final simplified tree.
    //! The node id  can be only refert to a Negated or a Joined Node
    NodeIdSet orphaned_nodes_;

    //! Used during construction of the simplified tree to map replaced nodes
    //! in the original tree to their new id in the simplified tree
    std::unordered_map<NodeId, MatchingNodes> node_ids_translation_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas