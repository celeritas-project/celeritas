//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgTreeUtils.hh
//! \brief Free functions to apply to a CSG tree
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "orange/OrangeTypes.hh"

#include "CsgTree.hh"
#include "CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{

//---------------------------------------------------------------------------//
/*!
 * Transformed CSG tree and mapping from the old one.
 *
 * The \c new_nodes member has the same size as the original tree, and is a map
 * from {old node ID} -> {equivalent simplified node ID}.
 */
using TransformedTree = std::pair<CsgTree, std::vector<NodeId>>;

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
[[nodiscard]] TransformedTree transform_negated_joins(CsgTree const& tree);

// Transform a CSG node into a string expression
[[nodiscard]] std::string
build_infix_string(CsgTree const& tree, orangeinp::NodeId n);

// Get the set of unsimplified surfaces in a tree
[[nodiscard]] std::vector<LocalSurfaceId> calc_surfaces(CsgTree const& tree);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
