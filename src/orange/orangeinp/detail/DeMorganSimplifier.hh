//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/DeMorganSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once
#include <vector>

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
 * This class serves as a helper for \c
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
    explicit DeMorganSimplifier(CsgTree const&);

    // Perform the simplification
    CsgTree operator()();

  private:
    //! Helper struct to translate ids from the original tree to ids in the
    //! simplified tree
    struct MatchingNodes
    {
        //! Set if a node has the exact same node in the simplified tree
        NodeId unmodified;

        //! Set if a node from the original tree redirects to a different node
        //! in the simplified tree. There are 3 possible redirections:
        //! 1. If the original node is a Joined node with a negated parent,
        //! redirect to the opposite join.
        //! 2. If the original node is a Negated node that should not be
        //! inserted in the simplified tree (happens when the only parent would
        //! be another Negated node) redirect to its children.
        //! 3. If the original node is a Negated node with a Joined child,
        //! redirect to the equivalent Join node in the simplified tree.
        //! If 2 and 3 are true, follows redirection in 2.
        //! If none of the above is true, this souldn't be set.

        // Indirections to new nodes
        // the negated node had a join child and now simplifies to that node
        NodeId simplified_to;
        // if a join node has been negated, this points to the opposite join
        NodeId opposite_join;
        // if this is deleted because of a double negation, point to the node
        // the double negation would simplify to
        NodeId double_negation_target;
        // if we inserted a new negation of that node
        NodeId new_negation;

        //! Whether any node id is set
        explicit operator bool() const noexcept
        {
            return simplified_to || opposite_join || double_negation_target
                   || new_negation || unmodified;
        }
    };

    // Row offset for the node in the parents_of matrix
    NodeId::size_type node_offset(NodeId) const;

    // First pass to find negated set operations
    void find_join_negations();

    // Declare negated nodes to add in the simplified tree
    void add_negation_for_operands(NodeId);

    // Second pass to build the simplified tree
    CsgTree build_simplified_tree();

    // Special handling for a Joined or Negated node
    bool process_negated_joined_nodes(NodeId, CsgTree&);

    // Check if this join node should be inserted in the simplified tree
    bool should_insert_join(NodeId);

    //! the tree to simplify
    CsgTree const& tree_;

    //! Set when we must insert a \c Negated parent for the given index
    std::vector<bool> new_negated_nodes_;

    //! Set when \c Joined nodes have a \c Negated parent, so we need to insert
    //! an opposite join node with negated operands
    std::vector<bool> negated_join_nodes_;

    //! Parent matrix. For nodes n1, n2, if n1 * tree_.size() + n2 is set, it
    //! means that n2 is a parent of n1
    std::vector<bool> parents_of;

    //! Used during construction of the simplified tree to map replaced nodes
    //! in the original tree to their new id in the simplified tree
    std::vector<MatchingNodes> node_ids_translation_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
