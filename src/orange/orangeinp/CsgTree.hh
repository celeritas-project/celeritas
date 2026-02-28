//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTree.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <optional>
#include <unordered_map>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/math/HashUtils.hh"

#include "CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
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
    using Node = orangeinp::Node;
    using NodeId = orangeinp::NodeId;
    using size_type = NodeId::size_type;
    using Insertion = std::pair<NodeId, bool>;
    using Simplification = std::optional<Node>;
    using VecNodeId = std::vector<NodeId>;
    //!@}

  public:
    // Construct with no nodes except "true" and "false"
    CsgTree();

    // Add a node and return the new ID and whether it's unique
    Insertion insert(Node&& n);

    // Add a surface ID and return the new ID
    inline Insertion insert(LocalSurfaceId s);

    //! Number of nodes
    NodeId::size_type size() const { return nodes_.size(); }

    // Find the node ID of the CSG expression if it exists
    NodeId find(Node&& n) const;

    // Get a node
    inline Node const& operator[](NodeId node_id) const;

    // Replace a node with a logically equivalent one, simplifying
    Node exchange(NodeId node_id, Node&& n);

    // Simplify a single node in-place [O(1)]
    Simplification simplify(NodeId);

    // Insert a new volume which root is the node id
    inline void insert_volume(NodeId);

    //! List of volumes defined in this tree
    VecNodeId const& volumes() const { return volumes_; }

    //// STATIC HELPERS ////

    //! Special ID of a node that's always 'true'
    static constexpr NodeId true_node_id() { return NodeId{0}; }
    //! Special ID of a node that's always 'false'
    static constexpr NodeId false_node_id() { return NodeId{1}; }

    //! Fast check for being a trivial node
    static constexpr bool is_boolean_node(NodeId n)
    {
        return n.unchecked_get() < 2;
    }

  private:
    // Tree structure: nodes may have been simplified
    std::vector<Node> nodes_;

    // Hashed simplified nodes, including eliminated but equivalent nodes
    std::unordered_map<Node, NodeId> ids_;

    // CSG node of each volume
    VecNodeId volumes_;

    //// HELPER FUNCTIONS ////

    // Get a simplified node expression
    bool simplify(Node& n) const;

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
auto CsgTree::insert(LocalSurfaceId s) -> Insertion
{
    return this->insert(orangeinp::Surface{s});
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
/*!
 * Insert a new volume which root is the node id.
 */
void CsgTree::insert_volume(NodeId node_id)
{
    CELER_EXPECT(node_id < nodes_.size());
    volumes_.push_back(std::move(node_id));
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
