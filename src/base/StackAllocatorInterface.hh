//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorInterface.hh
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
    //!@{
    //! Type aliases
    using size_type  = unsigned int;
    using value_type = T;
    //!@}

    Span<T>    storage;        //!< Allocated capacity
    size_type* size = nullptr; //!< Stored size

    // Whether the interface is initialized
    explicit inline CELER_FUNCTION operator bool() const;

    //! Total capacity of stack
    CELER_FUNCTION size_type capacity() const { return storage.size(); }
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
    CELER_EXPECT(storage.empty() || size);
    return !storage.empty();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
