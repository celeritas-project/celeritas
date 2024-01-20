//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
 * The tree is a topologically sorted graph where the leaves are the lower node
 * indices. The intent is for higher nodes to be simplified to boolean values,
 * e.g., by representing that all points in space are inside a cylinder by
 * replacing a \c Joined node with \c True . To facilitate this while
 * preserving node indices, \c True and \c False nodes are automatically
 * inserted as the first two node IDs. Because \c False is represented as "not
 * true" during evaluation, it actually stores \c Negated{true_node_id()} and
 * the deduplication table maps \c False to \c false_node_id() .
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

    // Replace a node with a logically equivalent one, simplifying
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

    // Get a node (only this class can modify the node once added)
    Node& at(NodeId node_id);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Print the tree's contents
std::ostream& operator<<(std::ostream& os, CsgTree const&);

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
