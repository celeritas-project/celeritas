//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/StackAllocatorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "Collection.hh"
#include "CollectionAlgorithms.hh"
#include "CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Storage for a stack and its dynamic size.
 */
template<class T, Ownership W, MemSpace M>
struct StackAllocatorData
{
    celeritas::Collection<T, W, M> storage;  //!< Allocated capacity
    celeritas::Collection<size_type, W, M> size;  //!< Stored size

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
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
        size = other.size;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize a stack allocator in host code.
 */
template<class T, MemSpace M>
inline void
resize(StackAllocatorData<T, Ownership::value, M>* data, size_type capacity)
{
    CELER_EXPECT(capacity > 0);
    resize(&data->storage, capacity);
    resize(&data->size, 1);
    celeritas::fill(size_type(0), &data->size);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
