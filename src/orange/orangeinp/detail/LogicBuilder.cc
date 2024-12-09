//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/LogicBuilder.cc
//---------------------------------------------------------------------------//
#include "LogicBuilder.hh"

#include <algorithm>
#include <type_traits>
#include <vector>

#include "corecel/math/Algorithms.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/CsgTree.hh"
#include "orange/orangeinp/detail/LogicBuilderPolicies.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert a single node to a given notation.
 *
 * The per-node local surfaces (faces) are sorted in ascending order of ID, not
 * of access, since they're always evaluated sequentially rather than as part
 * of the logic evaluation itself.
 */
template<class LogicBuilderPolicy>
auto LogicBuilder::operator()(NodeId n) const -> result_type
{
    CELER_EXPECT(n < tree_.size());
    static_assert(std::is_invocable_v<LogicBuilderPolicy, NodeId>);

    // Construct logic vector as local surface IDs
    VecLogic lgc;
    LogicBuilderPolicy build_impl{tree_, mapping_, &lgc};
    build_impl(n);

    // Construct sorted vector of faces
    std::vector<LocalSurfaceId> faces;
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

    return {std::move(faces), std::move(lgc)};
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template auto
LogicBuilder::operator()<InfixLogicBuilderPolicy>(NodeId) const -> result_type;
template auto
LogicBuilder::operator()<PostfixLogicBuilderPolicy>(NodeId) const -> result_type;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
