//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ProtoInterface.cc
//---------------------------------------------------------------------------//
#include "ProtoInterface.hh"

#include <deque>
#include <iterator>
#include <unordered_set>

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct a breadth-first ordering of protos.
 *
 * The input "global" universe will always be at the top of the list. Universes
 * may only depend on a universe with a larger ID.
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
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
