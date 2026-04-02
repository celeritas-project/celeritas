//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/LdgTraits.hh
//! \sa corecel/data/Ldg.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the arithmetic data for use with \c __ldg .
 *
 * Overload for arithmetic types: returns the pointer unchanged.
 *
 * To extend \c ldg support to a new type, define a free function
 * \c ldg_data(MyType const*) in the namespace of \c MyType (enabling
 * ADL-based lookup). The function must return a \c const pointer to an
 * arithmetic type.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION std::enable_if_t<std::is_arithmetic_v<T>, T const*>
ldg_data(T const* ptr) noexcept
{
    return ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Get a pointer to the underlying integer for an enum type.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION
    std::enable_if_t<std::is_enum_v<T>, std::underlying_type_t<T> const*>
    ldg_data(T const* ptr) noexcept
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<std::underlying_type_t<T> const*>(ptr);
}

//---------------------------------------------------------------------------//
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, class = void>
struct IsLdgSupported : std::false_type
{
};

template<class T>
struct IsLdgSupported<T, std::void_t<decltype(ldg_data(std::declval<T const*>()))>>
    : std::true_type
{
};

//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
//! Whether a type is supported by \c ldg
template<class T>
inline constexpr bool is_ldg_supported_v = detail::IsLdgSupported<T>::value;

//---------------------------------------------------------------------------//
}  // namespace celeritas
