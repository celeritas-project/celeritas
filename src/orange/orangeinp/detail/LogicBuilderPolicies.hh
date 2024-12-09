//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicBuilderPolicies.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "corecel/cont/VariantUtils.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class CsgTree;
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with postfix operation.
 *
 * This is an implementation detail of the \c PostfixLogicBuilder class. The
 * user invokes this class with a node ID (usually representing a cell), and
 * then this class recurses into the daughters using a tree visitor.
 * Example: \verbatim
    all(1, 3, 5) -> {{1, 3, 5}, "0 1 & 2 & &"}
    all(1, 3, !all(2, 4)) -> {{1, 2, 3, 4}, "0 2 & 1 3 & ~ &"}
 */
class PostfixLogicBuilderPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<logic_int>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

    static_assert(std::is_same_v<LocalSurfaceId::size_type, logic_int>,
                  "unsupported: add enum logic conversion for different-sized "
                  "face and surface ints");

  public:
    // Construct with optional mapping and logic vector to append to
    PostfixLogicBuilderPolicy(CsgTree const& tree,
                              VecSurface const* vs,
                              VecLogic* logic);

    //! Build from a node ID
    void operator()(NodeId const& n);

    //!@{
    //! \name Visit a node directly
    // Append 'true'
    void operator()(True const&);
    // False is never explicitly part of the node tree
    void operator()(False const&);
    // Append a surface ID
    void operator()(Surface const&);
    // Aliased nodes should never be reachable explicitly
    void operator()(Aliased const&);
    // Visit a negated node and append 'not'
    void operator()(Negated const&);
    // Visit daughter nodes and append the conjunction.
    void operator()(Joined const&);
    //!@}

  protected:
    VecLogic& logic() { return *logic_; }
    ContainerVisitor<CsgTree const&, NodeId>& visite_node()
    {
        return visit_node_;
    }

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
    VecSurface const* mapping_;
    VecLogic* logic_;
};

//---------------------------------------------------------------------------//
/*!
 * Recursively construct a logic vector from a node with infix operation.
 *
 * This is an implementation detail of the \c InfixLogicBuilder class. The
 * user invokes this class with a node ID (usually representing a cell), and
 * then this class recurses into the daughters using a tree visitor.
 */
class InfixLogicBuilderPolicy : public PostfixLogicBuilderPolicy
{
  public:
    //!@{
    //! \name Type aliases
    using PostfixLogicBuilderPolicy::VecLogic;
    using PostfixLogicBuilderPolicy::VecSurface;
    //!@}

  public:
    using PostfixLogicBuilderPolicy::PostfixLogicBuilderPolicy;

    //! Build from a node ID
    void operator()(NodeId const& n);

    //!@{
    //! \name Visit a node directly
    using PostfixLogicBuilderPolicy::operator();
    // Aliased nodes should never be reachable explicitly
    void operator()(Aliased const&);
    // Visit a negated node and append 'not'
    void operator()(Negated const&);
    // Visit daughter nodes and append the conjunction.
    void operator()(Joined const&);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
