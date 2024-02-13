//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/NodeSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/VariantUtils.hh"

#include "../CsgTree.hh"
#include "../CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Simplify a node by visiting up to one level below.
 *
 * \return The simplified node, an alias, or no_simplification
 */
class NodeSimplifier
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = NodeId::size_type;
    //!@}

  public:
    //! Sentinel for no simplification taking place
    static constexpr Aliased no_simplification() { return Aliased{NodeId{}}; }

    // Construct with the tree to visit
    explicit NodeSimplifier(CsgTree const& tree);

    // Replace an aliased node
    Node operator()(Aliased const& a) const;

    // Replace a negated node
    Node operator()(Negated const& n) const;

    // Replace joined node
    Node operator()(Joined& j) const;

    //! Other nodes don't simplify
    template<class T>
    Node operator()(T const&) const
    {
        return no_simplification();
    }

  private:
    ContainerVisitor<CsgTree const&, NodeId> visit_node_;
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
