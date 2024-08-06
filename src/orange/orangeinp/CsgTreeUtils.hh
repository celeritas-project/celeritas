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
std::vector<NodeId> replace_and_simplify(CsgTree* tree,
                                         orangeinp::NodeId n,
                                         orangeinp::Node replacement);

// Simplify the tree by sweeping
orangeinp::NodeId simplify_up(CsgTree* tree, orangeinp::NodeId start);

// Simplify the tree iteratively
void simplify(CsgTree* tree, orangeinp::NodeId start);

// Replace ~&(xs...) with |(~xs...) and ~|(xs...) with &(~xs...)
[[nodiscard]] CsgTree transform_negated_joins(CsgTree const& tree);

// Transform a CSG node into a string expression
[[nodiscard]] std::string
build_infix_string(CsgTree const& tree, orangeinp::NodeId n);

// Get the set of unsimplified surfaces in a tree
[[nodiscard]] std::vector<LocalSurfaceId> calc_surfaces(CsgTree const& tree);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
