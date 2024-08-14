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
 * celeritas::orangeinp::transform_negated_joins implementation. It applies
 * DeMorgan's law on a \c CsgTree so that all negations of a set operator are
 * removed and transformed into their equivalent operation.
 *
 * The simplification preserves subtrees referred to by \c CsgTree::volumes
 *
 * Instances of this class shouldn't outlive the \c CsgTree used to construct
 * it as we keep a reference to it.
 *
 * It is currently single-use: calling operator() twice on the same instance
 * isn't supported.
 */
class DeMorganSimplifier
{
  public:
    //! Construct a simplifier for the given tree
    explicit DeMorganSimplifier(CsgTree const& tree) : tree_(tree) {}

    // Perform the simplification
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

    // Unflag a node an its descendent previously marked as orphan
    void record_parent_for(NodeId);

    // Declare a Negated node with a Joined child
    void record_join_negation(NodeId);

    // Declare negated nodes to add in the simplified tree
    void add_negation_for_operands(NodeId);

    // Special handling for a Joined or Negated node
    bool process_negated_joined_nodes(NodeId, CsgTree&);

    // First pass to find negated set operations
    void find_join_negations();

    // Second pass to build the simplified tree
    CsgTree build_simplified_tree();

    //! the tree to simplify
    CsgTree const& tree_;

    //! For each node_id of these node ids, we must create a parent \c Negated
    NodeIdSet new_negated_nodes_;

    //! These \c Joined nodes have a \c Negated parent, so we need to insert an
    //! opposite join node with negated operands. The value is the node
    //! in tree_ referred to by the key.
    CachedNodeMap<Joined> negated_join_nodes_;

    //! Contains the node_id of \c Negated nodes with a \c Joined child. These
    //! nodes don't need to be inserted in the new tree and their parent can be
    //! redirected to the newly inserted opposite join. The value is the node
    //! in tree_ referred to by the key.
    CachedNodeMap<Negated> simplified_negated_nodes_;

    //! An orphan node should not be present in the final simplified tree.
    //! The node id can only be referring to a Negated or a Joined Node
    //! Other nodes should never become Orphanes
    NodeIdSet orphaned_nodes_;

    //! Used during construction of the simplified tree to map replaced nodes
    //! in the original tree to their new id in the simplified tree
    std::unordered_map<NodeId, MatchingNodes> node_ids_translation_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas