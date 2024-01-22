//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/CollectionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#ifndef CELER_DEVICE_COMPILE
#    include <vector>

#    include "../DeviceVector.hh"
#endif

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/LdgIterator.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "corecel/data/detail/LdgIteratorImpl.hh"
#include "corecel/sys/Device.hh"

#include "../Copier.hh"
#include "DisabledStorage.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, Ownership W, MemSpace M, typename = void>
struct CollectionTraits
{
    using type = T;
    using const_type = T const;
    using reference_type = type&;
    using const_reference_type = const_type&;
    using SpanT = Span<type>;
    using SpanConstT = Span<const_type>;
};

//---------------------------------------------------------------------------//
template<class T, MemSpace M>
struct CollectionTraits<T, Ownership::reference, M, void>
{
    using type = T;
    using const_type = T;
    using reference_type = type&;
    using const_reference_type = const_type&;
    using SpanT = Span<type>;
    using SpanConstT = Span<const_type>;
};

//---------------------------------------------------------------------------//
template<class T, MemSpace M>
struct CollectionTraits<T,
                        Ownership::const_reference,
                        M,
                        std::enable_if_t<!is_ldg_supported_v<std::add_const_t<T>>>>
{
    using type = T const;
    using const_type = T const;
    using reference_type = type&;
    using const_reference_type = const_type&;
    using SpanT = Span<type>;
    using SpanConstT = Span<const_type>;
};

//---------------------------------------------------------------------------//
template<class T, MemSpace M>
struct CollectionTraits<T,
                        Ownership::const_reference,
                        M,
                        std::enable_if_t<is_ldg_supported_v<std::add_const_t<T>>>>
{
    using type = T const;
    using const_type = T const;
    using reference_type = type&;
    using const_reference_type = const_type&;
    using SpanT = Span<type>;
    using SpanConstT = Span<const_type>;
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T,
                        Ownership::const_reference,
                        MemSpace::device,
                        std::enable_if_t<is_ldg_supported_v<std::add_const_t<T>>>>
{
    using type = T const;
    using const_type = T const;
    using reference_type = type;
    using const_reference_type = const_type;
    using SpanT = LdgSpan<const_type>;
    using SpanConstT = LdgSpan<const_type>;
};

//---------------------------------------------------------------------------//
//! Memspace-dependent storage for a collection
template<class T, Ownership W, MemSpace M>
struct CollectionStorage
{
    using type = typename CollectionTraits<T, W, M>::SpanT;
    type data;
};

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

//! Storage implementation for mapped host/device data
template<class T>
struct CollectionStorage<T, Ownership::value, MemSpace::mapped>
{
    static_assert(!std::is_same<T, bool>::value,
                  "bool is not compatible between vector and anything else");
#ifdef CELER_DEVICE_COMPILE
    // Use "not implemented" but __host__ __device__ decorated functions when
    // compiling in CUDA
    using type = DisabledStorage<T>;
#else
    using type = std::vector<T, PinnedAllocator<T>>;
#endif
    type data;
};

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
//! Assignment semantics for copying to mapped memory
template<>
struct CollectionAssigner<Ownership::value, MemSpace::mapped>
{
    CollectionAssigner()
    {
        CELER_VALIDATE(celeritas::device().can_map_host_memory(),
                       << "Device " << celeritas::device().device_id()
                       << " doesn't support unified addressing");
    }

    template<class T, Ownership W2, MemSpace M2>
    auto operator()(CollectionStorage<T, W2, M2> const& source)
        -> CollectionStorage<T, Ownership::value, M2>
    {
        return {{source.data.data(), source.data.data() + source.data.size()}};
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
