//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "TypeTraits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Wrap the low-level CUDA/HIP "load global memory" function.
 *
 * This low-level capability allows improved caching because we're \em
 * promising that no other thread can modify its value while the kernel is
 * active.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T ldg(T const* ptr)
{
    static_assert(std::is_arithmetic_v<T>,
                  "Only const arithmetic types are supported by __ldg");
#if CELER_DEVICE_COMPILE
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Reads a value T using __ldg builtin and return a copy of it
 */
template<class T, typename = void>
struct LdgLoader
{
    static_assert(std::is_const_v<T>, "Only const data are supported by __ldg");

    using value_type = std::remove_const_t<T>;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
        return ldg(p);
    }
};

/*!
 * Specialization when T == OpaqueId.
 * Wraps the underlying index in a OpaqueId when returning it.
 */
template<class I, class T>
struct LdgLoader<OpaqueId<I, T> const, void>
{
    using value_type = OpaqueId<I, T>;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
        return value_type{ldg(p->data())};
    }
};

/*!
 * Specialization when T == Quantity.
 * Wraps the underlying value in a Quantity when returning it.
 */
template<class I, class T>
struct LdgLoader<Quantity<I, T> const, void>
{
    using value_type = Quantity<I, T>;
    using pointer = std::add_pointer_t<value_type const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
        return value_type{ldg(p->data())};
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
        return value_type{ldg(reinterpret_cast<underlying_type const*>(p))};
#else
        return *p;
#endif
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
