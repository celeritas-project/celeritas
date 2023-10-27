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

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Reads a value T using __ldg builtin and return a copy of it
 */
template<class T>
struct LdgLoader
{
    static_assert(std::is_arithmetic_v<T> && std::is_const_v<T>,
                  "Only const arithmetic types are supported by __ldg");
    using value_type = T;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::remove_const_t<value_type>;

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
struct LdgLoader<OpaqueId<I, T> const>
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

// True if T is supported by a LdgLoader specialization
template<class T>
inline constexpr bool is_ldg_supported_v
    = std::is_const_v<T> && (std::is_arithmetic_v<T> || is_opaque_id_v<T>);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas