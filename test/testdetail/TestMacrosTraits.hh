//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/TestMacrosTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

namespace celeritas
{
namespace testdetail
{
//---------------------------------------------------------------------------//

//! Whether a type is a ref result
template<class T, class = void>
struct IsRefResult : std::false_type
{
};

template<class T>
struct IsRefResult<T, std::void_t<decltype(T::class_str[0])>>
    : std::bool_constant<std::is_same_v<decltype(T::class_str[0]), char const&>>
{
};

//! Whether a type is a reference result
template<class T>
inline constexpr bool IsRefResult_v = IsRefResult<T>::value;

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
