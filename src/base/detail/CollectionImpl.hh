//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CollectionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/Types.hh"

#ifndef __CUDA_ARCH__
#    include <vector>
#    include "base/Assert.hh"
#    include "base/DeviceVector.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, Ownership W>
struct CollectionTraits
{
    using SpanT                = Span<T>;
    using SpanConstT           = Span<const T>;
    using pointer              = T*;
    using const_pointer        = const T*;
    using reference_type       = T&;
    using const_reference_type = const T&;
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::reference>
{
    using SpanT                = Span<T>;
    using SpanConstT           = Span<T>;
    using pointer              = T*;
    using const_pointer        = T*;
    using reference_type       = T&;
    using const_reference_type = T&;
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::const_reference>
{
    using SpanT                = Span<const T>;
    using SpanConstT           = Span<const T>;
    using pointer              = const T*;
    using const_pointer        = const T*;
    using reference_type       = const T&;
    using const_reference_type = const T&;
};

//---------------------------------------------------------------------------//
//! Memspace-dependent storage for a collection
template<class T, Ownership W, MemSpace M>
struct CollectionStorage
{
    using type = typename CollectionTraits<T, W>::SpanT;
    type data;
};

template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::host>;
template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
//! Assignment semantics for a collection
template<Ownership W, MemSpace M>
struct CollectionAssigner
{
    template<class T, Ownership W2, MemSpace M2>
    CollectionStorage<T, W, M>
    operator()(const CollectionStorage<T, W2, M2>& source)
    {
        static_assert(W != Ownership::reference || W2 == W,
                      "Can't create a reference from a const reference");
        static_assert(M == M2,
                      "Collection assignment from a different memory space");
        return {{source.data.data(), source.data.size()}};
    }

    template<class T, Ownership W2, MemSpace M2>
    CollectionStorage<T, W, M> operator()(CollectionStorage<T, W2, M2>& source)
    {
        static_assert(M == M2,
                      "Collection assignment from a different memory space");
        static_assert(
            !(W == Ownership::reference && W2 == Ownership::const_reference),
            "Can't create a reference from a const reference");
        return {{source.data.data(), source.data.size()}};
    }
};

template<>
struct CollectionAssigner<Ownership::value, MemSpace::host>;
template<>
struct CollectionAssigner<Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
//! Check that sizes are acceptable when creating references from values
template<Ownership W>
struct CollectionStorageValidator
{
    template<class Size, class OtherSize>
    void operator()(Size, OtherSize)
    {
        /* No validation needed */
    }
};

template<>
struct CollectionStorageValidator<Ownership::value>
{
    template<class Size, class OtherSize>
    void operator()(Size dst, OtherSize src)
    {
        CELER_VALIDATE(dst == src,
                       << "collection is too large (" << sizeof(Size)
                       << "-byte int cannot hold " << src << " elements)");
    }
};

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
//! Storage implementation for managed host data
template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::host>
{
    using type = std::vector<T>;
    type data;
};

//! Storage implementation for managed device data
template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::device>
{
    using type = DeviceVector<T>;
    type data;
};

//---------------------------------------------------------------------------//
//! Assignment semantics for copying to host memory
template<>
struct CollectionAssigner<Ownership::value, MemSpace::host>
{
    template<class T, Ownership W2, MemSpace M2>
    CollectionStorage<T, Ownership::value, MemSpace::host>
    operator()(const CollectionStorage<T, W2, M2>& source)
    {
        static_assert(M2 == MemSpace::host,
                      "Can only assign host values from host data");
        return {{source.data.data(), source.data.data() + source.data.size()}};
    }
};

//---------------------------------------------------------------------------//
//! Assignment semantics for copying to device memory
template<>
struct CollectionAssigner<Ownership::value, MemSpace::device>
{
    template<class T, Ownership W2, MemSpace M2>
    CollectionStorage<T, Ownership::value, MemSpace::device>
    operator()(const CollectionStorage<T, W2, M2>& source)
    {
        static_assert(M2 == MemSpace::host,
                      "Can only assign by value from host to device");

        CollectionStorage<T, Ownership::value, MemSpace::device> result{
            DeviceVector<T>(source.data.size())};
        result.data.copy_to_device({source.data.data(), source.data.size()});
        return result;
    }
};

#else
// Give a useful error message when trying to use "value" collections when
// building device code
template<class T, MemSpace M>
struct CollectionStorage<T, Ownership::value, M>
{
    static_assert(sizeof(T) == 0,
                  "Value collections cannot be used from the NVCC device "
                  "compilation phase");
};

template<MemSpace M>
struct CollectionAssigner<Ownership::value, M>
{
    static_assert(static_cast<int>(M) == -1,
                  "Collections cannot be assigned from the NVCC device "
                  "compilation phase");
};

#endif

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
