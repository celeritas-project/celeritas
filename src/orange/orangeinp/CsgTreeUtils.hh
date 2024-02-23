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

// Convert a node to postfix notation
[[nodiscard]] std::vector<LocalSurfaceId::size_type>
build_postfix(CsgTree const& tree, orangeinp::NodeId n);

// Transform a CSG node into a string expression
[[nodiscard]] std::string
build_infix_string(CsgTree const& tree, orangeinp::NodeId n);

// Get the set of unsimplified surfaces in a tree
[[nodiscard]] std::vector<LocalSurfaceId> calc_surfaces(CsgTree const& tree);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
