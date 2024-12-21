//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"
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
 * The result is a pair of vectors: the sorted surface IDs comprising the faces
 * of this volume, and the logical representation using \em face IDs, i.e. with
 * the surfaces remapped to index of the surface in the face vector.
 *
 * The function is templated on a policy class that determines the logic
 * representation. The policy class must have an operator() that takes a
 * NodeId.
 *
 * The per-node local surfaces (faces) are sorted in ascending order of ID, not
 * of access, since they're always evaluated sequentially rather than as part
 * of the logic evaluation itself.
 */
template<class LogicBuilderPolicy>
inline result_type build_logic_repr(LogicBuilderPolicy&& policy, NodeId n)
{
    static_assert(std::is_invocable_v<LogicBuilderPolicy, NodeId>);
    static_assert(std::is_rvalue_reference_v<LogicBuilderPolicy&&>,
                  "Will move from policy: rvalue ref expected");

    // Construct logic vector as local surface IDs
    auto& lgc = policy.logic();
    CELER_EXPECT(lgc.empty());
    policy(n);

    // Construct sorted vector of faces
    VecSurface faces;
    for (auto const& v : lgc)
    {
        if (!logic::is_operator_token(v))
        {
            faces.push_back(LocalSurfaceId{v});
        }
    }

    // Sort and uniquify the vector
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());

    // Remap logic
    for (auto& v : lgc)
    {
        if (!logic::is_operator_token(v))
        {
            auto iter
                = find_sorted(faces.begin(), faces.end(), LocalSurfaceId{v});
            CELER_ASSUME(iter != faces.end());
            v = iter - faces.begin();
        }
    }
    return {faces, std::move(lgc)};
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
