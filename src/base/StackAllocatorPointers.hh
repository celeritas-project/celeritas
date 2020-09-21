//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Macros.hh"
#include "Span.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Pointers to stack allocator data.
 */
template<class T>
struct StackAllocatorPointers
{
    //@{
    //! Type aliases
    using size_type  = ull_int;
    using value_type = T;
    //@}

    span<T>    storage;        //!< Allocated capacity
    size_type* size = nullptr; //!< Stored size

    // Whether the interface is initialized
    explicit inline CELER_FUNCTION operator bool() const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the interface is initialized.
 */
template<class T>
CELER_FUNCTION StackAllocatorPointers<T>::operator bool() const
{
    REQUIRE(storage.empty() || size);
    return !storage.empty();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
