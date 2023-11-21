//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/LdgIteratorImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Reads a value T using __ldg builtin and return a copy of it
 */
template<class T, typename = void>
struct LdgLoader
{
    static_assert(std::is_arithmetic_v<T> && std::is_const_v<T>,
                  "Only const arithmetic types are supported by __ldg");
    using value_type = std::remove_const_t<T>;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
#if CELER_DEVICE_COMPILE
        return __ldg(p);
#else
        return *p;
#endif
    }
};

/*!
 * Specialization when T == OpaqueId.
 * Wraps the underlying index in a OpaqueId when returning it.
 */
template<class I, class T>
struct LdgLoader<OpaqueId<I, T> const, void>
{
    static_assert(std::is_arithmetic_v<T>,
                  "OpaqueId needs to be indexed with a type supported by "
                  "__ldg");
    using value_type = OpaqueId<I, T>;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
#if CELER_DEVICE_COMPILE
        return value_type{__ldg(&p->value_)};
#else
        return value_type{p->value_};
#endif
    }
};

/*!
 * Specialization when T == Quantity.
 * Wraps the underlying value in a Quantity when returning it.
 */
template<class I, class T>
struct LdgLoader<Quantity<I, T> const, void>
{
    static_assert(std::is_arithmetic_v<T>,
                  "Quantity needs to be represented by a type supported by "
                  "__ldg");
    using value_type = Quantity<I, T>;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
#if CELER_DEVICE_COMPILE
        return value_type{__ldg(&p->value_)};
#else
        return value_type{p->value_};
#endif
    }
};

template<class T>
struct LdgLoader<T const, std::enable_if_t<std::is_enum_v<T>>>
{
    using value_type = T;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;
    using underlying_type = std::underlying_type_t<T>;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
#if CELER_DEVICE_COMPILE
        // Technically breaks aliasing rule but it's not an issue:
        // the compiler doesn't derive any optimization and the pointer doesn't
        // escape the function
        return value_type{__ldg(reinterpret_cast<underlying_type const*>(p))};
#else
        return *p;
#endif
    }
};

// True if T is supported by a LdgLoader specialization
template<class T>
inline constexpr bool is_ldg_supported_v
    = std::is_const_v<T>
      && (std::is_arithmetic_v<T> || is_opaque_id_v<T> || is_quantity_v<T>
          || std::is_enum_v<T>);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas