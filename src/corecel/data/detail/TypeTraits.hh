//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/TypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class UnitT, class ValueT>
class Quantity;

namespace detail
{
//---------------------------------------------------------------------------//
//! Template matching to determine if T is a Quantity
template<class T>
struct IsQuantity : std::false_type
{
};
template<class V, class S>
struct IsQuantity<Quantity<V, S>> : std::true_type
{
};
template<class V, class S>
struct IsQuantity<Quantity<V, S> const> : std::true_type
{
};

//---------------------------------------------------------------------------//
//! True if T is a Quantity
template<class T>
inline constexpr bool is_quantity_v = IsQuantity<T>::value;

//---------------------------------------------------------------------------//
//! True if T is supported by a LdgLoader specialization
template<class T>
inline constexpr bool is_ldg_supported_v
    = std::is_const_v<T>
      && (std::is_arithmetic_v<T> || is_opaque_id_v<T> || is_quantity_v<T>
          || std::is_enum_v<T>);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
