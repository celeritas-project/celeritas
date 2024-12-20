//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <vector>

#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
using VecLogic = std::vector<logic_int>;
using VecSurface = std::vector<LocalSurfaceId>;
using result_type = std::pair<VecSurface, VecLogic>;

//---------------------------------------------------------------------------//
/*!
 * Construct a logic representation of a node.
 *
 * The optional surface mapping is an ordered vector of *existing* surface IDs.
 * Those surface IDs will be replaced by the index in the array. All existing
 * surface IDs must be present!
 *
 * The result is a pair of vectors: the sorted surface IDs comprising the faces
 * of this volume, and the logical representation using \em face IDs, i.e. with
 * the surfaces remapped to index of the surface in the face vector.
 *
 * The function is templated on a policy class that determines the logic
 * representation. The policy class must have an operator() that takes a NodeId
 * and be constructible from a CsgTree, VecSurface const*, and VecLogic*.
 */
template<class LogicBuilderPolicy>
result_type build_logic_repr(LogicBuilderPolicy&& policy, NodeId n);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
