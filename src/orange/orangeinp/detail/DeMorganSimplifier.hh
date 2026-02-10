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
 * The simplification preserves subtrees referred to by \c CsgTree::volumes .
 *
 * Instances of this class shouldn't outlive the \c CsgTree used to construct
 * it as we keep a reference to it.
 *
 * It is currently single-use: calling operator() twice on the same instance
 * isn't supported.
 *
 * The \c CsgTree being transformed should \em not have double negations
 * (the tree's insertion method will have simplified those).
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
    //// TYPES ////

    using MapNodeVecNode = std::unordered_map<NodeId, std::vector<NodeId>>;
    using SetNode = std::unordered_set<NodeId>;
    using MapNodeSetNode = std::unordered_map<NodeId, SetNode>;

    //! First meaningful node id in a CsgTree
    static constexpr auto first_node_id_{NodeId{2}};

    //! Helper struct to translate ids from the original tree to ids in the
    //! simplified tree
    // TODO: are more than one of these ever set simultaneously?
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

    //// HELPER FUNCTIONS ////

    // Get a non-aliased Node variant from the original tree
    Node const& get_node(NodeId) const;

    // First pass to find negated set operations
    void find_join_negations();

    // Record a parent-child relationship
    bool insert_parent(NodeId parent, NodeId child);

    // Declare negated nodes to add in the simplified tree
    bool insert_negated_children(NodeId);

    // Whether a child has any parent
    bool is_parent_of(NodeId parent, NodeId child) const;

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

    //// DATA ////

    //! the tree to simplify
    CsgTree const& tree_;

    //! Set when we must insert a \c Negated parent for the given index
    SetNode new_negated_nodes_;

    //! Set when \c Joined nodes have a \c Negated parent, so we need to insert
    //! an opposite join node with negated operands
    SetNode negated_join_nodes_;

    //! Parents matrix (original tree)
    // If parents_[c].count(p), p is parent of c
    MapNodeSetNode parents_;

    //! Whether the index is a volume in the original tree
    std::vector<bool> is_volume_node_;

    //! Map old node ID -> new node IDs:
    //! Used during construction of the simplified tree to map replaced nodes
    //! in the original tree to their new id in the simplified tree
    std::vector<MatchingNodes> matching_nodes_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
