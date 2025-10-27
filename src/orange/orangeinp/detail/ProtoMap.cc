//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/ProtoMap.cc
//---------------------------------------------------------------------------//
#include "ProtoMap.hh"

#include <deque>
#include <iterator>
#include <unordered_set>

#include "corecel/Assert.hh"

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
ProtoMap::ProtoMap(ProtoInterface const& global)
    : protos_{build_ordering(global)}
{
    univ_ids_.reserve(protos_.size());
    for (auto univ_id : range(UnivId{this->size()}))
    {
        ProtoInterface const* p = this->at(univ_id);
        CELER_ASSERT(p);
        auto&& [iter, inserted] = univ_ids_.insert({p, univ_id});
        CELER_ASSERT(inserted);
    }
    CELER_ENSURE(univ_ids_.size() == protos_.size());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
