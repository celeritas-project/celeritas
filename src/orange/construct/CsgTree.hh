//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/CsgTree.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <unordered_map>
#include <variant>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/math/HashUtils.hh"

#include "CsgTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * DAG of constructed CSG nodes within a universe.
 *
 * Inserting a new node performs one local simplification for "join" types
 * (removing duplicates, replacing zero- and one- entry special cases); and all
 * nodes are inserted into a deduplicated map of nodes that may be further
 * simplified later (see \c CsgAlgorithms).
 *
 * \note The node classes use aggregate initialization so cannot be created
 * directly as \c Node variant classes using \c std::in_place_type .
 */
class CsgTree
{
  public:
    //!@{
    //! \name Type aliases
    using Node = csg::Node;
    using NodeId = csg::NodeId;
    using size_type = NodeId::size_type;
    //!@}

  public:
    // Construct with no nodes except "true" and "false"
    CsgTree();

    // Add a node and return the new ID
    NodeId insert(Node&& n);

    // Add a surface ID and return the new ID
    inline NodeId insert(LocalSurfaceId s);

    //! Number of nodes
    size_type size() const { return nodes_.size(); }

    // Get a node
    inline Node const& operator[](NodeId node_id) const;

    // Aliased a node, simplifying, and returning original node
    Node exchange(NodeId node_id, Node&& n);

    // Simplify a single node in-place [O(1)]
    bool simplify(NodeId);

    //// STATIC HELPERS ////

    static constexpr NodeId true_node_id() { return NodeId{0}; }
    static constexpr NodeId false_node_id() { return NodeId{1}; }

  private:
    // Tree structure: nodes may have been simplified
    std::vector<Node> nodes_;

    // Hashed nodes, both original and simplified
    std::unordered_map<Node, NodeId> ids_;

    //// HELPER FUNCTIONS ////

    // Get a node (only this class can modify the node once added
    Node& at(NodeId node_id);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Print the tree's contents
std::ostream& operator<<(std::ostream& os, CsgTree const&);

// Replace a node in the tree with a boolean constant
csg::NodeId replace_down(CsgTree* tree, csg::NodeId n, csg::Node repl_node);

// Simplify the tree by sweeping
csg::NodeId simplify_up(CsgTree* tree, csg::NodeId start);

// Simplify the tree iteratively
void simplify(CsgTree* tree, csg::NodeId start);

// Convert a node to postfix notation
std::vector<LocalSurfaceId::size_type>
build_postfix(CsgTree const& tree, csg::NodeId n);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Add a surface.
 */
auto CsgTree::insert(LocalSurfaceId s) -> NodeId
{
    return this->insert(csg::Surface{s});
}

//---------------------------------------------------------------------------//
/*!
 * Get a node by ID.
 */
auto CsgTree::operator[](NodeId node_id) const -> Node const&
{
    CELER_EXPECT(node_id < nodes_.size());
    return nodes_[node_id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
