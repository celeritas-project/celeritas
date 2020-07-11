//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorView.hh
//---------------------------------------------------------------------------//
#ifndef base_StackAllocatorView_hh
#define base_StackAllocatorView_hh

#include "Assert.hh"
#include "Types.hh"
#include "Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reference data owned by a StackAllocatorContainer for use in StackAllocator.
 */
struct StackAllocatorView
{
    //! Size type needed for CUDA atomics compatibility
    using size_type = unsigned long long int;

    span<byte> storage;
    size_type* size;

    //! Check whether the view is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        REQUIRE(this->valid());
        return !storage.empty();
    }

    inline CELER_FUNCTION bool valid() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "StackAllocatorView.i.hh"

#endif // base_StackAllocatorView_hh
