//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/GeoSetup.cc
//---------------------------------------------------------------------------//
#include "GeoSetup.hh"

#include <deque>
#include <iterator>
#include <unordered_set>

#include "../ProtoInterface.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Construct a breadth-first ordering of protos.
 */
std::vector<ProtoInterface const*> build_ordering(ProtoInterface const& global)
{
    std::unordered_set<ProtoInterface const*> visited;
    std::vector<ProtoInterface const*> result;
    std::deque<ProtoInterface const*> stack{&global};

    // First get a depth-first ordering of daughters
    while (!stack.empty())
    {
        // Move front of stack to back of result
        ProtoInterface const* p = stack.front();
        CELER_ASSERT(p);
        stack.pop_front();

        // Mark as visited
        if (visited.insert(p).second)
        {
            // First time visitor: add to end of result, add daughters to end
            // of stack
            result.push_back(p);
            auto&& daughters = p->daughters();
            stack.insert(stack.end(), daughters.begin(), daughters.end());
        }
    }
    CELER_ENSURE(!result.empty());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with global proto for ordering.
 */
GeoSetup::GeoSetup(Tol const& tol, ProtoInterface const& global)
    : tol_{tol}, protos_{build_ordering(global)}
{
    CELER_EXPECT(tol_);

    uids_.reserve(protos_.size());
    for (auto uid : range(UniverseId{this->size()}))
    {
        ProtoInterface const* p = this->at(uid);
        CELER_ASSERT(p);
        auto&& [iter, inserted] = uids_.insert({p, uid});
        CELER_ASSERT(inserted);
    }
    CELER_ENSURE(uids_.size() == protos_.size());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
