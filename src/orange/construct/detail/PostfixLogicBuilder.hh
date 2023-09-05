//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
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
    // Construct with reference to vector to append to
    explicit inline PostfixLogicBuilder(CsgTree const& tree, VecLogic* logic);

    //! Build from a node ID
    inline void operator()(NodeId const& n);

    // Visit an actual surface
    inline void operator()(True const&);
    inline void operator()(False const&);
    inline void operator()(Surface const&);
    inline void operator()(Aliased const&);
    inline void operator()(Negated const&);
    inline void operator()(Joined const&);

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecLogic* logic_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to the logic expression.
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
    return this->visit_node_(*this, n);
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
    return logic_->push_back(s.id.unchecked_get());
}

//---------------------------------------------------------------------------//
/*!
 * Push an aliased node.
 *
 * TODO: aliased node shouldn't be reachable if we're fully simplified.
 */
void PostfixLogicBuilder::operator()(Aliased const& n)
{
    // CELER_ASSERT_UNREACHABLE();
    return (*this)(n.node);
}

//---------------------------------------------------------------------------//
/*!
 * If replacing a queued node with a boolean, ours should match.
 */
void PostfixLogicBuilder::operator()(Negated const& n)
{
    (*this)(n.node);
    logic_->push_back(logic::lnot);
}

//---------------------------------------------------------------------------//
/*!
 * If replacing a queued node with a boolean, ours should match.
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
