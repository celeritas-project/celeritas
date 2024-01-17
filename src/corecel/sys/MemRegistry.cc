//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MemRegistry.cc
//---------------------------------------------------------------------------//
#include "MemRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create a new entry and push it onto the stack.
 */
MemUsageId MemRegistry::push()
{
    // Add a new entry
    MemUsageId result_id(entries_.size());
    entries_.resize(entries_.size() + 1);

    // Record the parent index and update the stack
    if (!stack_.empty())
    {
        entries_.back().parent_index = stack_.back();
    }
    stack_.push_back(result_id);

    return result_id;
}

//---------------------------------------------------------------------------//
/*!
 * Pop the last entry.
 */
void MemRegistry::pop()
{
    CELER_EXPECT(!stack_.empty());
    stack_.pop_back();
}

//---------------------------------------------------------------------------//
// Globally shared registry of memory usage
MemRegistry& mem_registry()
{
    static MemRegistry mr;
    return mr;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
