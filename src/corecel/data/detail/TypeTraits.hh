//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/TypeTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class ValueT, class SizeT>
class OpaqueId;

template<class UnitT, class ValueT>
class Quantity;

namespace detail
{
//---------------------------------------------------------------------------//
//! Template matching to determine if T is an OpaqueId
template<class T>
struct IsOpaqueId : std::false_type
{
};

template<class V, class S>
struct IsOpaqueId<OpaqueId<V, S>> : std::true_type
{
};

template<class V, class S>
struct IsOpaqueId<OpaqueId<V, S> const> : std::true_type
{
};

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
//! True if T is an OpaqueID
template<class T>
inline constexpr bool is_opaque_id_v = IsOpaqueId<T>::value;

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
