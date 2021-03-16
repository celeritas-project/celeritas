//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Collection.hh"
#include "Macros.hh"
#include "Types.hh"

#ifndef __CUDA_ARCH__
#    include "CollectionAlgorithms.hh"
#    include "CollectionBuilder.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Storage for a stack and its dynamic size.
 */
template<class T, Ownership W, MemSpace M>
struct StackAllocatorData
{
    celeritas::Collection<T, W, M>         storage; //!< Allocated capacity
    celeritas::Collection<size_type, W, M> size;    //!< Stored size

    // Whether the interface is initialized
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !storage.empty() && !size.empty();
    }

    //! Total capacity of stack
    CELER_FUNCTION size_type capacity() const { return storage.size(); }

    //! Assign from another stack
    template<Ownership W2, MemSpace M2>
    StackAllocatorData& operator=(StackAllocatorData<T, W2, M2>& other)
    {
        CELER_EXPECT(other);
        storage = other.storage;
        size    = other.size;
        return *this;
    }
};

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Resize a stack allocator in host code.
 */
template<class T, MemSpace M>
inline void
resize(StackAllocatorData<T, Ownership::value, M>* data, size_type capacity)
{
    CELER_EXPECT(capacity > 0);
    make_builder(&data->storage).resize(capacity);
    make_builder(&data->size).resize(1);
    celeritas::fill(size_type(0), &data->size);
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
