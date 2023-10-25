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
    static_assert(std::is_arithmetic_v<T>);
    using value_type = T;
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
struct LdgLoader<OpaqueId<I, T>>
{
    using value_type = OpaqueId<I, T>;
    using pointer = std::add_pointer_t<T const>;
    using reference = value_type;

    CELER_CONSTEXPR_FUNCTION static reference read(pointer p)
    {
#if CELER_DEVICE_COMPILE
        return value_type{__ldg(p)};
#else
        return value_type{*p};
#endif
    }
};
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas