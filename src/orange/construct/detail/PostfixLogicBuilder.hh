//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/detail/PostfixLogicBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/VariantUtils.hh"
#include "orange/OrangeTypes.hh"

#include "../CsgTree.hh"

namespace celeritas
{
namespace csg
{
//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with postfix operation.
 *
 * This is an implementation detail of the \c build_postfix function. The user
 * invokes this class with a node ID (usually representing a cell), and then
 * this class recurses into the daughters using a tree visitor.
 */
class PostfixLogicBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    //!@}

    static_assert(std::is_same_v<LocalSurfaceId::size_type, logic_int>,
                  "unsupported: add enum logic conversion for different-sized "
                  "face and surface ints");

  public:
    // Construct with pointer to vector to append to
    explicit inline PostfixLogicBuilder(CsgTree const& tree, VecLogic* logic);

    //! Build from a node ID
    inline void operator()(NodeId const& n);

    //!@{
    //! \name Visit a node directly
    // Append 'true'
    inline void operator()(True const&);
    // False is never explicitly part of the node tree
    inline void operator()(False const&);
    // Append a surface ID
    inline void operator()(Surface const&);
    // Aliased nodes should never be reachable explicitly
    inline void operator()(Aliased const&);
    // Visit a negated node and append 'not'
    inline void operator()(Negated const&);
    // Visit daughter nodes and append the conjunction.
    inline void operator()(Joined const&);
    //!@}

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecLogic* logic_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with pointer to the logic expression.
 */
PostfixLogicBuilder::PostfixLogicBuilder(CsgTree const& tree, VecLogic* logic)
    : visit_node_{tree}, logic_{logic}
{
    CELER_EXPECT(logic_);
}

//---------------------------------------------------------------------------//
/*!
 * Build from a node ID.
 */
void PostfixLogicBuilder::operator()(NodeId const& n)
{
    visit_node_(*this, n);
}

//---------------------------------------------------------------------------//
/*!
 * Append the "true" token.
 */
void PostfixLogicBuilder::operator()(True const&)
{
    logic_->push_back(logic::ltrue);
}

//---------------------------------------------------------------------------//
/*!
 * Explicit "False" should never be possible for a CSG cell.
 *
 * The 'false' standin is always aliased to "not true" in the CSG tree.
 */
void PostfixLogicBuilder::operator()(False const&)
{
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Push a surface ID.
 */
void PostfixLogicBuilder::operator()(Surface const& s)
{
    CELER_EXPECT(s.id < logic::lbegin);
    logic_->push_back(s.id.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Push an aliased node.
 *
 * TODO: aliased node shouldn't be reachable if we're fully simplified.
 */
void PostfixLogicBuilder::operator()(Aliased const& n)
{
    (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * Visit a negated node and append 'not'.
 */
void PostfixLogicBuilder::operator()(Negated const& n)
{
    (*this)(n.node);
    logic_->push_back(logic::lnot);
}

//---------------------------------------------------------------------------//
/*!
 * Visit daughter nodes and append the conjunction.
 */
void PostfixLogicBuilder::operator()(Joined const& n)
{
    CELER_EXPECT(n.nodes.size() > 1);

    // Visit first node, then add conjunction for subsequent nodes
    auto iter = n.nodes.begin();
    (*this)(*iter++);

    while (iter != n.nodes.end())
    {
        (*this)(*iter++);
        logic_->push_back(n.op);
    }
}

//---------------------------------------------------------------------------//
}  // namespace csg
}  // namespace celeritas
