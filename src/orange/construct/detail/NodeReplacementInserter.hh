//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/detail/NodeReplacementInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>
#include <vector>

#include "../CsgTree.hh"
#include "../CsgTypes.hh"

namespace celeritas
{
namespace csg
{
//---------------------------------------------------------------------------//
/*!
 * Add a node ID and its "replaced" value.
 *
 * This implementation detail of the "replace down" algorithm adds daughters
 * to a queue of nodes to visit, along with their replacement values.
 */
class NodeReplacementInserter
{
  public:
    //!@{
    //! \name Type aliases
    using VecNode = std::vector<std::pair<NodeId, Node>>;
    //!@}

  public:
    // Construct with pointers to the stack and replacement
    inline NodeReplacementInserter(VecNode* stack, Node const& repl);

    // Simplify node types that reference other nodes
    inline void operator()(Aliased const& n);
    inline void operator()(Negated const& n);
    inline void operator()(Joined const& n);

    // Check that replacement matches our stored boolean
    inline void operator()(True const&);
    inline void operator()(False const&);

    // Surfaces cannot be simplified further
    void operator()(Surface const&) {}

  private:
    VecNode* stack_;
    Node const& repl_;
    Node negated_;
    OperatorToken join_token_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to the stack and the replacement value.
 *
 * For now the replacement must be a boolean.
 */
NodeReplacementInserter::NodeReplacementInserter(VecNode* stack,
                                                 Node const& repl)
    : stack_{stack}, repl_{repl}
{
    CELER_EXPECT(stack_);
    CELER_EXPECT(is_boolean_node(repl_));

    // Save "negated" node and "join" implication for daughters
    if (std::holds_alternative<True>(repl_))
    {
        negated_ = Node{False{}};
        join_token_ = op_and;  // all{...} = true
    }
    else
    {
        negated_ = Node{True{}};
        join_token_ = op_or;  // any{...} = false
    }
}

//---------------------------------------------------------------------------//
/*!
 * Check that the replacement node matches this queued node.
 */
void NodeReplacementInserter::operator()(True const&)
{
    CELER_ASSERT(std::holds_alternative<True>(repl_));
}

//---------------------------------------------------------------------------//
/*!
 * Check that the replacement node matches this queued node.
 */
void NodeReplacementInserter::operator()(False const&)
{
    CELER_ASSERT(std::holds_alternative<False>(repl_));
}

//---------------------------------------------------------------------------//
/*!
 * Push the target of an aliased node onto the stack.
 *
 * Aliasing a node implies the alias has the same value.
 */
void NodeReplacementInserter::operator()(Aliased const& n)
{
    stack_->emplace_back(n.node, repl_);
}

//---------------------------------------------------------------------------//
/*!
 * Push a negated node onto the stack.
 *
 * Negating a node implies its daughter has the opposite value.
 */
void NodeReplacementInserter::operator()(Negated const& n)
{
    stack_->emplace_back(n.node, negated_);
}

//---------------------------------------------------------------------------//
/*!
 * Some 'join' operations imply requirements for the daughters.
 *
 * If this node is "true" and it uses an "and" operation, all daughters must be
 * true. Likewise, "false" with "or" implies all daughters are false.
 */
void NodeReplacementInserter::operator()(Joined const& n)
{
    if (n.op == join_token_)
    {
        for (NodeId d : n.nodes)
        {
            stack_->emplace_back(d, repl_);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace csg
}  // namespace celeritas
