//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/LdgTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Traits for extracting underlying type and pointer for __ldg operations.
 *
 * Specialize this class on the base type, not a const type.
 */
template<class T, class = void>
struct LdgTraits
{
    using underlying_type = void;
    static CELER_CONSTEXPR_FUNCTION underlying_type const* data(T const*)
    {
        static_assert(sizeof(T) == 0,
                      "Only arithmetic underlying types can be loaded with "
                      "LDG");
        return {};
    }
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for arithmetic types.
 */
template<class T>
struct LdgTraits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
{
    using underlying_type = T;

    static CELER_CONSTEXPR_FUNCTION underlying_type const* data(T const* ptr)
    {
        return ptr;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Specialization for enum types.
 */
template<class T>
struct LdgTraits<T, std::enable_if_t<std::is_enum_v<T>>>
{
    using underlying_type = std::underlying_type_t<T>;

    static CELER_CONSTEXPR_FUNCTION underlying_type const* data(T const* ptr)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<underlying_type const*>(ptr);
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
