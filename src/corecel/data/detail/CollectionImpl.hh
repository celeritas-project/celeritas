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
template<class T>
struct DefaultCollectionTraits
{
    using type = T;
    using const_type = T const;
    using SpanT = Span<type>;
    using SpanConstT = Span<const_type>;
    using StorageT = SpanT;
};

//---------------------------------------------------------------------------//
template<class T, Ownership W, MemSpace M>
struct CollectionTraits : DefaultCollectionTraits<T>
{
};

//---------------------------------------------------------------------------//
template<class T, MemSpace M>
struct CollectionTraits<T, Ownership::reference, M> : DefaultCollectionTraits<T>
{
    using const_type = T;  //!< Return type is *mutable* for reference!
    using SpanConstT = Span<T>;
};

//---------------------------------------------------------------------------//
template<class T, MemSpace M>
struct CollectionTraits<T, Ownership::const_reference, M>
    : DefaultCollectionTraits<T>
{
    using type = T const;
    using SpanT = AutoLdgSpan<M, T const>;
    using SpanConstT = SpanT;
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::value, MemSpace::host>
    : DefaultCollectionTraits<T>
{
    static_assert(!std::is_same<T, bool>::value,
                  "bool is not compatible between vector and anything else");

#ifdef CELER_DEVICE_COMPILE
    using StorageT = DisabledStorage<T>;
#else
    using StorageT = std::vector<T>;
#endif
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::value, MemSpace::device>
    : DefaultCollectionTraits<T>
{
#ifdef CELER_DEVICE_COMPILE
    using StorageT = DisabledStorage<T>;
#else
    using StorageT = DeviceVector<T>;
#endif
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::value, MemSpace::mapped>
    : DefaultCollectionTraits<T>
{
    static_assert(!std::is_same<T, bool>::value,
                  "bool is not compatible between vector and anything else");
#ifdef CELER_DEVICE_COMPILE
    using StorageT = DisabledStorage<T>;
#else
    using StorageT = std::vector<T, PinnedAllocator<T>>;
#endif
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
template<class T, Ownership SW, MemSpace SM, Ownership DW, MemSpace DM>
void copy_collection(Span<T const> src,
                     typename CollectionTraits<T, DW, DM>::StorageT* dst)
{
    using DstStorageT = CollectionTraits<T, DW, DM>::StorageT;

    // Const cast is OK because the only time it's used is when this is called
    // with Ownership::reference and the caller is doing T* -> const T*
    auto* data = const_cast<T*>(src.data());
    auto size = src.size();

    if constexpr (DW == Ownership::value && DM == MemSpace::mapped)
    {
        CELER_VALIDATE(celeritas::device().can_map_host_memory(),
                       << "device " << celeritas::device().device_id()
                       << " doesn't support unified addressing");
    }
    if constexpr (DW == Ownership::value && DM == SM)
    {
        // Allocate and copy at the same time: destination "owns" the memory
        dst->assign(data, data + size);
    }
    else if constexpr (DM == SM)
    {
        // Copy pointers in same memspace, prohibiting const violation
        static_assert(
            !(SW == Ownership::const_reference && DW == Ownership::reference),
            "cannot assign from const reference to reference");

        *dst = DstStorageT{data, size};
    }
    else
    {
        if constexpr (DW == Ownership::value)
        {
            // Allocate destination
            *dst = DstStorageT(size);
        }

        CELER_VALIDATE(dst->size() == size,
                       << "collection assignment from " << to_cstring(SM)
                       << " to " << to_cstring(DM)
                       << " failed: cannot copy from source size " << size
                       << " to destination size " << dst->size());

        // Copy across memory boundary
        Copier<T, DM> copy_to_dst{{dst->data(), dst->size()}};
        copy_to_dst(SM, {data, size});
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
