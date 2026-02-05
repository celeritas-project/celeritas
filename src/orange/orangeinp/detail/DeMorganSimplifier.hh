//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/DeMorganSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "corecel/math/HashUtils.hh"

#include "../CsgTree.hh"
#include "../CsgTypes.hh"

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
 *
 * The \c CsgTree being simplified shouldn't contain alias nodes or double
 * negations.
 */
class DeMorganSimplifier
{
  public:
    using TransformedTree = std::pair<CsgTree, std::vector<NodeId>>;

    // Construct a simplifier for the given tree
    explicit DeMorganSimplifier(CsgTree const&);

    // Perform the simplification
    TransformedTree operator()();

  private:
    using NodePair = std::pair<NodeId, NodeId>;

    struct NodePairHash
    {
        std::size_t operator()(NodePair const& p) const noexcept
        {
            return hash_combine(p.first, p.second);
        }
    };

    using SparseMatrix2D = std::unordered_set<NodePair, NodePairHash>;
    using NodeSet = std::unordered_set<NodeId>;

    //! CsgTree node 0 is always True{} and can't be the parent of any node
    //! so reuse that bit to tell that a given node is a volume
    static constexpr auto is_volume_index_{NodeId{0}};
    //! CsgTree node 1 is always a Negated node parent of node 0, so we can
    //! reuse that bit to tell if a node has a parent as it's never set for
    //! node id >= 2
    static constexpr auto has_parents_index_{NodeId{1}};

    //! First meaningful node id in a CsgTree
    static constexpr auto first_node_id_{NodeId{2}};

    //! Helper struct to translate ids from the original tree to ids in the
    //! simplified tree
    struct MatchingNodes
    {
        //! Set if a node has the exact same node in the simplified tree
        NodeId unmodified;

        //! Indirections to new simplified join following DeMorgan's law
        //! Set if the original node is a negated node.
        NodeId simplified_to;
        //! If a join node has been negated, this points to the opposite join
        //! Set if the original node is a joined node.
        NodeId opposite_join;

        //! Set if we need to insert a new negation of that node
        //! Set if the original node is a leaf node.
        NodeId new_negation;

        //! Whether any matching node id is set
        explicit operator bool() const noexcept
        {
            return unmodified || simplified_to || opposite_join || new_negation;
        }

        // Lookup a node an equal node in the simplified tree
        NodeId equivalent_node() const;
    };

    using NodeMap = std::unordered_map<NodeId, MatchingNodes>;

    // Dereference Aliased nodes
    NodeId dealias(NodeId) const;

    // First pass to find negated set operations
    void find_join_negations();

    // Declare negated nodes to add in the simplified tree
    void add_negation_for_operands(NodeId);

    // Second pass to build the simplified tree
    CsgTree build_simplified_tree();

    // Special handling for a Joined or Negated node
    bool process_negated_joined_nodes(NodeId, CsgTree&);

    // Create an opposite Joined node
    Joined build_negated_node(Joined const&);

    // Check if this join node should be inserted in the simplified tree
    bool should_insert_join(NodeId);

    // Access or create a translation entry
    MatchingNodes& translation(NodeId);
    // Find a translation entry, or nullptr if none exist
    MatchingNodes const* find_translation(NodeId) const;

    //! the tree to simplify
    CsgTree const& tree_;

    //! Set when we must insert a \c Negated parent for the given index
    NodeSet new_negated_nodes_;

    //! Set when \c Joined nodes have a \c Negated parent, so we need to insert
    //! an opposite join node with negated operands
    NodeSet negated_join_nodes_;

    //! Parents matrix. If the pair {e1, e2} exists, e2 is parent of e1
    SparseMatrix2D parents_;

    //! Used during construction of the simplified tree to map replaced nodes
    //! in the original tree to their new id in the simplified tree
    NodeMap node_ids_translation_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
