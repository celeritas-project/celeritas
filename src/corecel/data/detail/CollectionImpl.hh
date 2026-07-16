//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/CollectionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#ifndef CELER_DEVICE_COMPILE
#    include <vector>
#endif

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/LdgSpan.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "corecel/sys/Device.hh"

#ifdef CELER_DEVICE_COMPILE
#    include "DisabledStorage.hh"
#else
#    include "corecel/data/DeviceVector.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Tag structure for accessing a memspace-local span of data
template<class T, MemSpace M>
struct AllItems_t
{
};

//---------------------------------------------------------------------------//
//! Memspace-safe pointer based on a container type
template<class Container, MemSpace M>
using ContainerObserverPtr
    = ObserverPtr<std::remove_pointer_t<typename Container::pointer>, M>;

//---------------------------------------------------------------------------//
//! Common traits for collections (unusual ones override these inherited types)
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
    using SpanT = AutoLdgSpan<T const>;
    using SpanConstT = SpanT;
    using StorageT = Span<T const>;
};

//---------------------------------------------------------------------------//
template<class T>
struct CollectionTraits<T, Ownership::value, MemSpace::host>
    : DefaultCollectionTraits<T>
{
    static_assert(!std::is_same_v<T, bool>,
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
template<Ownership W, class Size, class OtherSize>
inline void validate_storage(Size dst, OtherSize src)
{
    if constexpr (W == Ownership::value)
    {
        CELER_VALIDATE(dst == src,
                       << "collection is too large (" << sizeof(Size)
                       << "-byte int cannot hold " << src << " elements)");
    }
}

//---------------------------------------------------------------------------//
//! Check that mappable memory is a valid destination
inline void validate_mappable_memory()
{
    auto& d = celeritas::device();
    CELER_VALIDATE(
        d.can_map_host_memory(),
        << "invalid collection MemSpace::mapped: device "
        << (d ? "does not support unified addressing" : "is not enabled"));
}

//! Check before assignment that the sizes are equal
inline void validate_compatible_size(
    std::size_t dst_size, MemSpace dm, std::size_t src_size, MemSpace sm)

{
    CELER_VALIDATE(dst_size == src_size,
                   << "collection assignment from " << to_cstring(sm) << " to "
                   << to_cstring(dm) << " failed: cannot copy from source size "
                   << src_size << " to destination size " << dst_size);
}

//---------------------------------------------------------------------------//
/*!
 * Copy-assign a collection via its storage.
 *
 * Since the copy operation is done only on the default stream, this should
 * only be performed during setup and during testing. State allocations should
 * use a separate resize+copy.
 */
template<class T, Ownership SW, MemSpace SM, Ownership DW, MemSpace DM>
inline void copy_collection(
    Span<T const> csrc, typename CollectionTraits<T, DW, DM>::StorageT* dst)
{
    CELER_EXPECT(dst);
    using DstStorageT = typename CollectionTraits<T, DW, DM>::StorageT;

    static_assert(
        !(SW == Ownership::const_reference && DW == Ownership::reference),
        "cannot assign from const reference to reference");

    // Const cast is OK: we need something like it because of the different
    // combinations of mutable/const operator=, and the static assertion above
    // ensures the correct semantics
    Span<T> src = {const_cast<T*>(csrc.data()), csrc.size()};

    if constexpr (DM == MemSpace::mapped)
    {
        validate_mappable_memory();
    }
    if constexpr (DM == SM)
    {
        // Copy/reference within the same memory space
        if constexpr (DW == Ownership::value)
        {
            // Allocate (if necessary) and copy to the new collection
            dst->assign(src.data(), src.data() + src.size());
        }
        else
        {
            // Make span in same memspace, prohibiting const violation
            *dst = DstStorageT{src.data(), src.size()};
        }
    }
    else
    {
        // Copy from one memspace to another
        if constexpr (DW == Ownership::value)
        {
            // Allocate destination
            *dst = DstStorageT(src.size());
        }

        if constexpr (!CELER_USE_DEVICE)
        {
            // Mark unreachable for optimization and coverage
            CELER_ASSERT_UNREACHABLE();
        }

        // When crossing memspace, we're copying the underlying data, *not*
        // copying pointers: so the destination "view" should be to some
        // already-allocated memory that we're copying to
        validate_compatible_size(dst->size(), DM, src.size(), SM);

        // Copy across memory boundary with raw pointer/size
        // (cannot use make_span since dst could be DeviceVector)
        Copier<T, DM> copy_to_dst{{dst->data(), dst->size()}};
        copy_to_dst(SM, src);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
