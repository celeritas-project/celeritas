//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/InvalidValueTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Array.hh"

#include "../Collection.hh"

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
 * types, and fill generic types with garbage values that look like
 * `0xd0d0d0d0`.
 */
template<class T, class Enable = void>
struct InvalidValueTraits
{
    static T value()
    {
        T result;
        std::memset(&result, 0xd0, sizeof(T)); // 4*b"\xf0\x9f\xa6\xa4".decode()
        return result;
    }
};

//---------------------------------------------------------------------------//
template<class I, class T>
struct InvalidValueTraits<OpaqueId<I, T>, void>
{
    static constexpr OpaqueId<I, T> value()
    {
        return OpaqueId<I, T>(std::numeric_limits<T>::max() / 2);
    }
};

template<class T>
struct InvalidValueTraits<T,
                          typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    static constexpr T value() { return std::numeric_limits<T>::max() / 2; }
};

template<class T, size_type N>
struct InvalidValueTraits<Array<T, N>, void>
{
    static Array<T, N> value()
    {
        Array<T, N> result;
        result.fill(InvalidValueTraits<T>::value());
        return result;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Fill a collection with an invalid value (host only).
 */
template<MemSpace M>
struct InvalidFiller
{
    template<class T>
    void operator()(T*)
    {
    }
};

template<>
struct InvalidFiller<MemSpace::host>
{
    template<class T, Ownership W, class I>
    void operator()(Collection<T, W, MemSpace::host, I>* c)
    {
        CELER_EXPECT(c);

        T    val   = InvalidValueTraits<T>::value();
        auto items = (*c)[AllItems<T>{}];
        std::fill(items.begin(), items.end(), val);
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
} // namespace detail
} // namespace celeritas
