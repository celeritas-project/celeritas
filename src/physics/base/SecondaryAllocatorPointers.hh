//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryAllocatorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * View to data for allocating secondaries.
 */
struct SecondaryAllocatorPointers
{
    //! Size type needed for CUDA atomics compatibility
    using size_type = unsigned long long int;

    span<Secondary> storage;        // View to storage space for secondaries
    size_type*      size = nullptr; // Total number of secondaries stored

    // Whether the pointers are assigned
    explicit inline CELER_FUNCTION operator bool() const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Check whether the pointers are assigned.
 */
CELER_FUNCTION SecondaryAllocatorPointers::operator bool() const
{
    REQUIRE(storage.empty() || size);
    return !storage.empty();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
