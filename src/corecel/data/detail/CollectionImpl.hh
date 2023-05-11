//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/CollectionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#ifndef CELER_DEVICE_COMPILE
#    include <vector>

#    include "../DeviceVector.hh"
#endif

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

#include "../Copier.hh"
#include "DisabledStorage.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, Ownership W>
struct CollectionTraits
{
    using type = T;
    using const_type = T const;
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::reference>
{
    using type = T;
    using const_type = T;
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::const_reference>
{
    using type = T const;
    using const_type = T const;
};

//---------------------------------------------------------------------------//
//! Memspace-dependent storage for a collection
template<class T, Ownership W, MemSpace M>
struct CollectionStorage
{
    using type = Span<typename CollectionTraits<T, W>::type>;
    type data;
};

template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::host>;
template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
//! Storage implementation for managed host data
template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::host>
{
    static_assert(!std::is_same<T, bool>::value,
                  "bool is not compatible between vector and anything else");
#ifdef CELER_DEVICE_COMPILE
    // Use "not implemented" but __host__ __device__ decorated functions when
    // compiling in CUDA
    using type = DisabledStorage<T>;
#else
    using type = std::vector<T>;
#endif
    type data;
};

//! Storage implementation for managed device data
template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::device>
{
#ifdef CELER_DEVICE_COMPILE
    // Use "not implemented" but __host__ __device__ decorated functions when
    // compiling in CUDA
    using type = DisabledStorage<T>;
#else
    using type = DeviceVector<T>;
#endif
    type data;
};

//---------------------------------------------------------------------------//
//! Assignment semantics for a collection
template<Ownership W, MemSpace M>
struct CollectionAssigner
{
    template<class T, Ownership W2, MemSpace M2>
    CollectionStorage<T, W, M>
    operator()(CollectionStorage<T, W2, M2> const& source)
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

//---------------------------------------------------------------------------//
//! Assignment semantics for copying to host memory
template<>
struct CollectionAssigner<Ownership::value, MemSpace::host>
{
    template<class T, Ownership W2>
    CollectionStorage<T, Ownership::value, MemSpace::host>
    operator()(CollectionStorage<T, W2, MemSpace::host> const& source)
    {
        return {{source.data.data(), source.data.data() + source.data.size()}};
    }

    template<class T, Ownership W2>
    CollectionStorage<T, Ownership::value, MemSpace::host>
    operator()(CollectionStorage<T, W2, MemSpace::device> const& source)
    {
        CollectionStorage<T, Ownership::value, MemSpace::host> result{
            std::vector<T>(source.data.size())};
        Copier<T, MemSpace::host> copy{
            {result.data.data(), result.data.size()}};
        copy(MemSpace::device, {source.data.data(), source.data.size()});
        return result;
    }
};

//---------------------------------------------------------------------------//
//! Assignment semantics for copying to device memory
template<>
struct CollectionAssigner<Ownership::value, MemSpace::device>
{
    template<class T>
    using StorageValDev
        = CollectionStorage<T, Ownership::value, MemSpace::device>;

    template<class T, Ownership W2, MemSpace M2>
    StorageValDev<T> operator()(CollectionStorage<T, W2, M2> const& source)
    {
        static_assert(M2 == MemSpace::host,
                      "Can only assign by value from host to device");

        StorageValDev<T> result{
            typename StorageValDev<T>::type(source.data.size())};
        result.data.copy_to_device({source.data.data(), source.data.size()});
        return result;
    }
};

//---------------------------------------------------------------------------//
//! Template matching to determine if T is an OpaqueId
template<class T>
struct IsOpaqueId
{
    static constexpr bool value = false;
};
template<class V, class S>
struct IsOpaqueId<OpaqueId<V, S>>
{
    static constexpr bool value = true;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
