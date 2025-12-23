//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/FillInvalid.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Return an 'invalid' value.
 *
 * This is used to reproducibly replicate construction on device, where
 * {cuda,hip}Malloc doesn't call the default constructors on data.
 *
 * Instead of assigning 'NaN', which may work automatically for sentinel logic
 * such as "valid if x > 0)", we assign large (half-max) values for numeric
 * types, and fill trivial types with garbage values that look like
 * `0xd0d0d0d0`.
 */
template<class T>
struct InvalidValueTraits
{
    static T value()
    {
        if constexpr (std::is_trivially_copyable_v<T>)
        {
            // Assigning garbage data to a result with a potentially
            // *non-trivial type*, such as a struct of OpaqueIds, even if it's
            // trivially copyable. However, this is in essence what's going on
            // when we allocate space for nontrivial types on device:
            // whatever's there (whether memset on NVIDIA or uninitialized on
            // AMD) is not going to have "placement new" applied since we're
            // not using thrust or calling Filler to launch initialization
            // kernels on all our datatypes. Reinterpret the data as bytes and
            // assign garbage values.
            T v;
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            std::memset(reinterpret_cast<unsigned char*>(&v), 0xd0, sizeof(T));
            return v;
        }
        else if constexpr (TriviallyCopyable_v<T>)
        {
            // The type is specialized by a developer as being "trivially
            // copyable" when it's technically not: examples are with the HIP
            // RNG and VecGeom nav state. Avoid breaking copy constructors.
            return T{};
        }
        else
        {
            static_assert(
                sizeof(T) == 0,
                R"(Filling can only be done to trivially copyable classes)");
        }
    }
};

//---------------------------------------------------------------------------//
template<class I, class T>
struct InvalidValueTraits<OpaqueId<I, T>>
{
    static constexpr OpaqueId<I, T> value()
    {
        return OpaqueId<I, T>(std::numeric_limits<T>::max() / 2);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Fill a collection with an invalid value (nullop if not host/mapped).
 */
template<MemSpace M>
struct InvalidFiller
{
    template<class T, Ownership W, class I>
    void operator()(Collection<T, W, M, I>* c)
    {
        CELER_EXPECT(c);

        T val = InvalidValueTraits<T>::value();
        auto items = (*c)[AllItems<T, M>{}];
        std::fill(items.begin(), items.end(), val);
    }
};

template<>
struct InvalidFiller<MemSpace::device>
{
    template<class T>
    void operator()(T*)
    {
        /* Null-op */
    }
};

//---------------------------------------------------------------------------//
/*!
 * Fill a collection with an invalid value (host only).
 *
 * This can probably be removed once we switch to C++17 and \c
 * CollectionBuilder::resize uses \code if constexpr \endcode .
 */
template<class T, Ownership W, MemSpace M, class I = ItemId<T>>
void fill_invalid(Collection<T, W, M, I>* c)
{
    return InvalidFiller<M>{}(c);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
