//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/CsgTreeUtils.hh
//! \brief Free functions to apply to a CSG tree
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "orange/OrangeTypes.hh"

#include "CsgTypes.hh"

namespace celeritas
{
class CsgTree;
//---------------------------------------------------------------------------//

// Replace a node in the tree with a boolean constant
csg::NodeId replace_down(CsgTree* tree, csg::NodeId n, csg::Node repl_node);

// Simplify the tree by sweeping
csg::NodeId simplify_up(CsgTree* tree, csg::NodeId start);

// Simplify the tree iteratively
void simplify(CsgTree* tree, csg::NodeId start);

// Convert a node to postfix notation
std::vector<LocalSurfaceId::size_type>
build_postfix(CsgTree const& tree, csg::NodeId n);

// Get the set of unsimplified surfaces in a tree
std::vector<LocalSurfaceId> calc_surfaces(CsgTree const& tree);

//---------------------------------------------------------------------------//
}  // namespace celeritas
