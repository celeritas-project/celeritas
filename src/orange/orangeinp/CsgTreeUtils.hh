//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeUtils.hh
//! \brief Free functions to apply to a CSG tree
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "orange/OrangeTypes.hh"

#include "CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class CsgTree;
//---------------------------------------------------------------------------//

// Replace a node in the tree with a boolean constant
orangeinp::NodeId
replace_down(CsgTree* tree, orangeinp::NodeId n, orangeinp::Node repl_node);

// Simplify the tree by sweeping
orangeinp::NodeId simplify_up(CsgTree* tree, orangeinp::NodeId start);

// Simplify the tree iteratively
void simplify(CsgTree* tree, orangeinp::NodeId start);

// Transform a CSG node into a string expression
[[nodiscard]] std::string
build_infix_string(CsgTree const& tree, orangeinp::NodeId n);

// Get the set of unsimplified surfaces in a tree
[[nodiscard]] std::vector<LocalSurfaceId> calc_surfaces(CsgTree const& tree);

//---------------------------------------------------------------------------//
/*!
 * Convert a node to postfix notation.
 *
 * The optional surface mapping is an ordered vector of *existing* surface IDs.
 * Those surface IDs will be replaced by the index in the array. All existing
 * surface IDs must be present!
 */
class PostfixLogicBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using VecLogic = std::vector<LocalSurfaceId::size_type>;
    using VecSurface = std::vector<LocalSurfaceId>;
    //!@}

  public:
    //! Construct from a tree
    explicit PostfixLogicBuilder(CsgTree const& tree) : tree_{tree} {}

    //! Construct from a tree and surface remapping
    PostfixLogicBuilder(CsgTree const& tree, VecSurface const& old_ids)
        : tree_{tree}, mapping_{&old_ids}
    {
    }

    // Convert a single node to postfix notation
    [[nodiscard]] VecLogic operator()(NodeId n) const;

  private:
    CsgTree const& tree_;
    VecSurface const* mapping_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
