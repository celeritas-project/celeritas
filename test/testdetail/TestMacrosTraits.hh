//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/TestMacrosTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <string_view>
#include <type_traits>
namespace celeritas
{
class Constant;
namespace testdetail
{
//---------------------------------------------------------------------------//
// CONTAINER TYPE
//---------------------------------------------------------------------------//

/*!
 * Check if type T is a container that needs recursive checking.
 */
template<class T, class = void>
struct IsContainer : std::false_type
{
};

template<>
struct IsContainer<std::string> : std::false_type
{
};

template<>
struct IsContainer<std::string_view> : std::false_type
{
};

template<class T>
struct IsContainer<T, std::void_t<typename T::const_iterator>> : std::true_type
{
};

template<class T, std::size_t N>
struct IsContainer<T[N]> : std::true_type
{
};

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

//! Whether a type is a container
template<class T>
inline constexpr bool IsContainer_v = IsContainer<T>::value;

//! Whether a type is a reference result
template<class T>
inline constexpr bool IsRefResult_v = IsRefResult<T>::value;

//---------------------------------------------------------------------------//
// COMMON PRECISION
//---------------------------------------------------------------------------//

/*!
 * Get a "least common denominator" for soft comparisons.
 */
template<class T, class U = T>
struct CommonPrecision
{
    using type = void;
};

template<class T>
struct CommonPrecision<T, T>
{
    using type = T;
};

#define CELERTEST_COMMON_PRECISION(TEMPLATE, T, U, V) \
    template<TEMPLATE> \
    struct CommonPrecision<T, U> : CommonPrecision<V> \
    { \
    }

// When comparing doubles to floats, use the floating point epsilon for
// comparison
CELERTEST_COMMON_PRECISION(, double, float, float);
CELERTEST_COMMON_PRECISION(, float, double, float);
// Allow reference type to be an int (i.e. user writes 0 or 1 instead of 0.)
CELERTEST_COMMON_PRECISION(class T, int, T, T);
// Allow reference or actual type to be a constant
CELERTEST_COMMON_PRECISION(class T, Constant, T, T);
CELERTEST_COMMON_PRECISION(class T, T, Constant, T);
CELERTEST_COMMON_PRECISION(, Constant, Constant, double);

template<class T, class U = T>
using CommonPrecision_t = typename CommonPrecision<T, U>::type;

//! Whether soft equivalence can be performed on the given types.
template<class T, class U>
inline constexpr bool can_soft_equiv = !std::is_void_v<CommonPrecision_t<T, U>>;

//---------------------------------------------------------------------------//
/*!
 * Get the type of a container.
 */
template<class T, class = void>
struct ValueType
{
    using type = typename T::value_type;
};

template<class T, std::size_t N>
struct ValueType<T[N]>
{
    using type = T;
};

template<class T>
using ValueType_t = typename ValueType<T>::type;

/*!
 * Recursively get the underlying scalar type of a container.
 */
template<class T, class = void>
struct ScalarValueType
{
    using type = T;
};

template<class T>
struct ScalarValueType<T, std::enable_if_t<IsContainer_v<T>>>
{
    using type = typename ScalarValueType<ValueType_t<T>>::type;
};

template<class T>
using ScalarValueType_t = typename ScalarValueType<T>::type;

//! Whether a container *holds* another container
template<class T>
inline constexpr bool IsNestedContainer_v = IsContainer_v<ValueType_t<T>>;

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
