//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/NodeReplacer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>
#include <ostream>
#include <variant>
#include <vector>

#include "corecel/io/EnumStringMapper.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Add a node ID and its "replaced" value.
 *
 * This implementation detail of the CSG simplify algorithm updates the
 * dependent logical state from a single node.
 *
 * \return whether a change to the state took place
 */
class NodeReplacer
{
  public:
    enum NodeRepl
    {
        unvisited,
        unknown,
        known_false,
        known_true
    };

    //!@{
    //! \name Type aliases
    using VecRepl = std::vector<NodeRepl>;
    //!@}

  public:
    // Construct with pointers to the state and the node being replaced
    inline NodeReplacer(VecRepl* state, NodeId n);

    // Simplify node types that reference other nodes
    inline bool operator()(Aliased const& n);
    inline bool operator()(Negated const& n);
    inline bool operator()(Joined const& n);

    // Check that replacement matches our stored boolean
    inline bool operator()(True const&);
    inline bool operator()(False const&);

    // Literals don't propagate information
    inline bool operator()(Surface const&) { return {}; }

  private:
    VecRepl* state_;
    NodeRepl repl_;

    // Update the given node to the replacement value
    inline bool update(NodeId n, NodeRepl repl);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to the state and the replacement value.
 *
 * For now the replacement must be a boolean.
 */
NodeReplacer::NodeReplacer(VecRepl* state, NodeId n) : state_{state}
{
    CELER_EXPECT(state_);
    repl_ = (*state_)[n.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Check that the replacement node matches this state.
 */
bool NodeReplacer::operator()(True const&)
{
    CELER_ASSERT(repl_ == NodeRepl::known_true);
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Check that the replacement node matches this state.
 */
bool NodeReplacer::operator()(False const&)
{
    CELER_ASSERT(repl_ == NodeRepl::known_false);
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Replace the target of an aliased node.
 *
 * Aliasing a node implies the alias has the same value.
 */
bool NodeReplacer::operator()(Aliased const& n)
{
    return this->update(n.node, repl_);
}

//---------------------------------------------------------------------------//
/*!
 * Update the target of a negated node.
 *
 * Negating a node implies its child has the opposite value.
 */
bool NodeReplacer::operator()(Negated const& n)
{
    NodeRepl repl = [r = repl_] {
        switch (r)
        {
            case NodeRepl::known_false:
                return NodeRepl::known_true;
            case NodeRepl::known_true:
                return NodeRepl::known_false;
            default:
                return r;
        }
    }();

    return this->update(n.node, repl);
}

//---------------------------------------------------------------------------//
/*!
 * Some 'join' operations imply requirements for the daughters.
 *
 * If this node is "true" and it uses an "and" operation, all daughters must be
 * true. Likewise, "false" with "or" implies all daughters are false.
 */
bool NodeReplacer::operator()(Joined const& n)
{
    NodeRepl repl = repl_;
    if ((repl == NodeRepl::known_true && n.op == op_or)
        || (repl == NodeRepl::known_false && n.op == op_and))
    {
        // ... but with this combination we can't prove
        // anything about the children
        repl = NodeRepl::unknown;
    }

    bool updated{false};
    for (NodeId d : n.nodes)
    {
        // NOTE: don't write as "updated || this->update(...)" dude to short
        // circuit logic
        bool daughter_updated = this->update(d, repl);
        updated = updated || daughter_updated;
    }
    return updated;
}

//---------------------------------------------------------------------------//
/*!
 * Update the given node to the replacement value.
 */
bool NodeReplacer::update(NodeId n, NodeRepl repl)
{
    CELER_EXPECT(n < state_->size());
    NodeRepl& dest = (*state_)[n.unchecked_get()];
    CELER_VALIDATE(
        !(dest == NodeRepl::known_true && repl == NodeRepl::known_false)
            && !(dest == NodeRepl::known_false && repl == NodeRepl::known_true),
        << "encountered logical contradiction in CSG operation");
    if (dest < repl)
    {
        dest = repl;
        return true;
    }
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
