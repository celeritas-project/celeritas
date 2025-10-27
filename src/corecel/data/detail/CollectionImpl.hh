//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/CollectionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#ifndef CELER_DEVICE_COMPILE
#    include <vector>

#    include "corecel/data/DeviceVector.hh"
#endif

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/LdgIterator.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "corecel/sys/Device.hh"

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

    inline static constexpr Ownership ownership = W;
    inline static constexpr MemSpace memspace = M;
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

    inline static constexpr Ownership ownership = Ownership::value;
    inline static constexpr MemSpace memspace = MemSpace::host;
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

    inline static constexpr Ownership ownership = Ownership::value;
    inline static constexpr MemSpace memspace = MemSpace::device;
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

    inline static constexpr Ownership ownership = Ownership::value;
    inline static constexpr MemSpace memspace = MemSpace::mapped;
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
/*!
 * Copy assign a collection via its storage.
 */
template<class S, class T, Ownership DW, MemSpace DM>
void copy_collection(S& src, CollectionStorage<T, DW, DM>* dst)
{
    constexpr MemSpace SM = std::remove_const_t<S>::memspace;
    using DstStorageT = typename CollectionStorage<T, DW, DM>::type;

    auto* data = src.data.data();
    size_type size = src.data.size();

    if constexpr (DW == Ownership::value && DM == MemSpace::mapped)
    {
        CELER_VALIDATE(celeritas::device().can_map_host_memory(),
                       << "device " << celeritas::device().device_id()
                       << " doesn't support unified addressing");
    }

    if constexpr (DW == Ownership::value && DM == SM)
    {
        // Allocate and copy at the same time: destination "owns" the memory
        dst->data.assign(data, data + size);
    }
    else if constexpr (DM == SM)
    {
        // Copy pointers in same memspace, prohibiting const violation
        constexpr Ownership SW = std::remove_const_t<S>::ownership;

        static_assert(
            !(SW == Ownership::const_reference && DW == Ownership::reference),
            "cannot assign from const reference to reference");

        dst->data = DstStorageT{data, size};
    }
    else
    {
        if constexpr (DW == Ownership::value)
        {
            // Allocate destination
            dst->data = DstStorageT(size);
        }

        CELER_VALIDATE(dst->data.size() == size,
                       << "collection assignment from " << to_cstring(SM)
                       << " to " << to_cstring(DM)
                       << " failed: cannot copy from source size " << size
                       << " to destination size " << dst->data.size());

        // Copy across memory boundary
        Copier<T, DM> copy_to_dst{{dst->data.data(), dst->data.size()}};
        copy_to_dst(SM, {data, size});
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
