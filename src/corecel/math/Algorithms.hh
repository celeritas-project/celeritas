//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Algorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

#include "detail/AlgorithmsImpl.hh"
#include "detail/MathImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Replace/extend <utility>
//---------------------------------------------------------------------------//
//! Implement perfect forwarding with device-friendly functions.
template<class T>
CELER_CONSTEXPR_FUNCTION T&&
forward(typename std::remove_reference<T>::type& v) noexcept
{
    return static_cast<T&&>(v);
}

//! \cond
template<class T>
CELER_CONSTEXPR_FUNCTION T&&
forward(typename std::remove_reference<T>::type&& v) noexcept
{
    return static_cast<T&&>(v);
}
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Cast a value as an rvalue reference to allow move construction.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION auto move(T&& v) noexcept ->
    typename std::remove_reference<T>::type&&
{
    return static_cast<typename std::remove_reference<T>::type&&>(v);
}

//---------------------------------------------------------------------------//
/*!
 * Support swapping of trivial types.
 */
template<class T>
CELER_FORCEINLINE_FUNCTION void trivial_swap(T& a, T& b) noexcept
{
    static_assert(std::is_trivially_move_constructible<T>::value,
                  "Value is not trivially copyable");
    static_assert(std::is_trivially_move_assignable<T>::value,
                  "Value is not trivially movable");
    static_assert(std::is_trivially_destructible<T>::value,
                  "Value is not trivially destructible");
    T temp{::celeritas::move(a)};
    a = ::celeritas::move(b);
    b = ::celeritas::move(temp);
}

//---------------------------------------------------------------------------//
/*!
 * Exchange values on host or device.
 */
template<class T, class U = T>
CELER_FORCEINLINE_FUNCTION T exchange(T& dst, U&& src)
{
    T orig = std::move(dst);
    dst = std::forward<U>(src);
    return orig;
}

//---------------------------------------------------------------------------//
// Replace/extend <functional>
//---------------------------------------------------------------------------//
/*!
 * Evaluator for the first argument being less than the second.
 */
template<class T = void>
struct Less
{
    CELER_CONSTEXPR_FUNCTION auto
    operator()(T const& lhs, T const& rhs) const noexcept -> decltype(auto)
    {
        return lhs < rhs;
    }
};

//! Specialization of less with template deduction
template<>
struct Less<void>
{
    template<class T, class U>
    CELER_CONSTEXPR_FUNCTION auto operator()(T&& lhs, U&& rhs) const
        -> decltype(auto)
    {
        return ::celeritas::forward<T>(lhs) < ::celeritas::forward<U>(rhs);
    }
};

//---------------------------------------------------------------------------//
// Replace/extend <algorithm>
//---------------------------------------------------------------------------//
/*!
 * Whether the predicate is true for all items.
 */
template<class InputIt, class Predicate>
inline CELER_FUNCTION bool all_of(InputIt iter, InputIt last, Predicate p)
{
    for (; iter != last; ++iter)
    {
        if (!p(*iter))
            return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the predicate is true for any item.
 */
template<class InputIt, class Predicate>
inline CELER_FUNCTION bool any_of(InputIt iter, InputIt last, Predicate p)
{
    for (; iter != last; ++iter)
    {
        if (p(*iter))
            return true;
    }
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Clamp the value between lo and hi values.
 *
 * If the value is between lo and hi, return the value. Otherwise, return lo if
 * it's below it, or hi above it.
 *
 * This replaces:
 * \code
   min(hi, max(lo, v))
 * \endcode
 * or
 * \code
   max(v, min(v, lo))
 * \endcode
 * assuming that the relationship between \c lo and \c hi holds.
 */
template<class T>
inline CELER_FUNCTION T const& clamp(T const& v, T const& lo, T const& hi)
{
    CELER_EXPECT(!(hi < lo));
    return v < lo ? lo : hi < v ? hi : v;
}

//---------------------------------------------------------------------------//
/*!
 * Return the value or (if it's negative) then zero.
 *
 * This is constructed to correctly propagate \c NaN.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T clamp_to_nonneg(T v) noexcept
{
    return (v < 0) ? 0 : v;
}

//---------------------------------------------------------------------------//
/*!
 * Find the insertion point for a value in a sorted list using a binary search.
 */
template<class ForwardIt, class T, class Compare>
CELER_FORCEINLINE_FUNCTION ForwardIt
lower_bound(ForwardIt first, ForwardIt last, T const& value, Compare comp)
{
    using CompareRef = std::add_lvalue_reference_t<Compare>;
    return ::celeritas::detail::lower_bound_impl<CompareRef>(
        first, last, value, comp);
}

//---------------------------------------------------------------------------//
/*!
 * Find the insertion point for a value in a sorted list using a binary search.
 */
template<class ForwardIt, class T>
CELER_FORCEINLINE_FUNCTION ForwardIt lower_bound(ForwardIt first,
                                                 ForwardIt last,
                                                 T const& value)
{
    return ::celeritas::lower_bound(first, last, value, Less<>{});
}

//---------------------------------------------------------------------------//
/*!
 * Find the insertion point for a value in a sorted list using a linear search.
 */
template<class ForwardIt, class T, class Compare>
CELER_FORCEINLINE_FUNCTION ForwardIt lower_bound_linear(ForwardIt first,
                                                        ForwardIt last,
                                                        T const& value,
                                                        Compare comp)
{
    using CompareRef = std::add_lvalue_reference_t<Compare>;
    return ::celeritas::detail::lower_bound_linear_impl<CompareRef>(
        first, last, value, comp);
}

//---------------------------------------------------------------------------//
/*!
 * Find the insertion point for a value in a sorted list using a linear search.
 */
template<class ForwardIt, class T>
CELER_FORCEINLINE_FUNCTION ForwardIt lower_bound_linear(ForwardIt first,
                                                        ForwardIt last,
                                                        T const& value)
{
    return ::celeritas::lower_bound_linear(first, last, value, Less<>{});
}

//---------------------------------------------------------------------------//
/*!
 * Find the first element which is greater than <value>
 */
template<class ForwardIt, class T, class Compare>
CELER_FORCEINLINE_FUNCTION ForwardIt
upper_bound(ForwardIt first, ForwardIt last, T const& value, Compare comp)
{
    using CompareRef = std::add_lvalue_reference_t<Compare>;
    return ::celeritas::detail::upper_bound_impl<CompareRef>(
        first, last, value, comp);
}

//---------------------------------------------------------------------------//
/*!
 * Find the first element which is greater than <value>
 */
template<class ForwardIt, class T>
CELER_FORCEINLINE_FUNCTION ForwardIt upper_bound(ForwardIt first,
                                                 ForwardIt last,
                                                 T const& value)
{
    return ::celeritas::upper_bound(first, last, value, Less<>{});
}

//---------------------------------------------------------------------------//
/*!
 * Partition elements in the given range, "true" before "false".
 *
 * This is done by swapping elements until the range is partitioned.
 */
template<class ForwardIt, class Predicate>
CELER_FORCEINLINE_FUNCTION ForwardIt partition(ForwardIt first,
                                               ForwardIt last,
                                               Predicate pred)
{
    using PredicateRef = std::add_lvalue_reference_t<Predicate>;
    return ::celeritas::detail::partition_impl<PredicateRef>(first, last, pred);
}

//---------------------------------------------------------------------------//
/*!
 * Sort an array on a single thread.
 *
 * This implementation is not thread-safe nor cooperative, but it can be called
 * from CUDA code.
 */
template<class RandomAccessIt, class Compare>
CELER_FORCEINLINE_FUNCTION void
sort(RandomAccessIt first, RandomAccessIt last, Compare comp)
{
    using CompareRef = std::add_lvalue_reference_t<Compare>;
    return ::celeritas::detail::heapsort_impl<CompareRef>(first, last, comp);
}

//---------------------------------------------------------------------------//
/*!
 * Sort an array on a single thread.
 */
template<class RandomAccessIt>
CELER_FORCEINLINE_FUNCTION void sort(RandomAccessIt first, RandomAccessIt last)
{
    ::celeritas::sort(first, last, Less<>{});
}

//---------------------------------------------------------------------------//
/*!
 * Return the higher of two values.
 *
 * This function is specialized when building CUDA device code, which has
 * special intrinsics for max.
 */
#ifndef __CUDA_ARCH__
template<class T>
#else
template<class T, typename = std::enable_if_t<!std::is_arithmetic<T>::value>>
#endif
CELER_CONSTEXPR_FUNCTION T const& max(T const& a, T const& b) noexcept
{
    return (b > a) ? b : a;
}

#ifdef __CUDA_ARCH__
template<class T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
CELER_CONSTEXPR_FUNCTION T max(T a, T b) noexcept
{
    return ::max(a, b);
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Return the lower of two values.
 *
 * This function is specialized when building CUDA device code, which has
 * special intrinsics for min.
 */
#ifndef __CUDA_ARCH__
template<class T>
#else
template<class T, typename = std::enable_if_t<!std::is_arithmetic<T>::value>>
#endif
CELER_CONSTEXPR_FUNCTION T const& min(T const& a, T const& b) noexcept
{
    return (b < a) ? b : a;
}

#ifdef __CUDA_ARCH__
template<class T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
CELER_CONSTEXPR_FUNCTION T min(T a, T b) noexcept
{
    return ::min(a, b);
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Return an iterator to the lowest value in the range as defined by Compare.
 */
template<class ForwardIt, class Compare>
inline CELER_FUNCTION ForwardIt min_element(ForwardIt iter,
                                            ForwardIt last,
                                            Compare comp)
{
    // Avoid incrementing past the end
    if (iter == last)
        return last;

    ForwardIt result = iter++;
    for (; iter != last; ++iter)
    {
        if (comp(*iter, *result))
            result = iter;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return an iterator to the lowest value in the range.
 */
template<class ForwardIt>
CELER_FORCEINLINE_FUNCTION ForwardIt min_element(ForwardIt first,
                                                 ForwardIt last)
{
    return ::celeritas::min_element(first, last, Less<decltype(*first)>{});
}

//---------------------------------------------------------------------------//
// Replace/extend <cmath>
//---------------------------------------------------------------------------//
//! Generate overloads for a single-argument math function
#define CELER_WRAP_MATH_FLOAT_DBL_1(PREFIX, FUNC)        \
    CELER_FORCEINLINE_FUNCTION float FUNC(float value)   \
    {                                                    \
        return ::PREFIX##FUNC##f(value);                 \
    }                                                    \
    CELER_FORCEINLINE_FUNCTION double FUNC(double value) \
    {                                                    \
        return ::PREFIX##FUNC(value);                    \
    }
#define CELER_WRAP_MATH_FLOAT_DBL_PTR_2(PREFIX, FUNC)                        \
    CELER_FORCEINLINE_FUNCTION void FUNC(float value, float* a, float* b)    \
    {                                                                        \
        return ::PREFIX##FUNC##f(value, a, b);                               \
    }                                                                        \
    CELER_FORCEINLINE_FUNCTION void FUNC(double value, double* a, double* b) \
    {                                                                        \
        return ::PREFIX##FUNC(value, a, b);                                  \
    }

//---------------------------------------------------------------------------//
/*!
 * Return an integer power of the input value.
 *
 * Example: \code
  assert(9.0 == ipow<2>(3.0));
  assert(256 == ipow<8>(2));
  static_assert(256 == ipow<8>(2));
 \endcode
 */
template<unsigned int N, class T>
CELER_CONSTEXPR_FUNCTION T ipow(T v) noexcept
{
    if constexpr (N == 0)
    {
        (void)sizeof(v);  // Suppress warning in older compilers
        return 1;
    }
    else if constexpr (N % 2 == 0)
    {
        return ipow<N / 2>(v) * ipow<N / 2>(v);
    }
    else
    {
        return v * ipow<(N - 1) / 2>(v) * ipow<(N - 1) / 2>(v);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Raise a number to a power with simplifying assumptions.
 *
 * This should be faster than `std::pow` because we don't worry about
 * exceptions for zeros, infinities, or negative values for a.
 *
 * Example: \code
  assert(9.0 == fastpow(3.0, 2.0));
 \endcode
 */
template<class T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
inline CELER_FUNCTION T fastpow(T a, T b)
{
    CELER_EXPECT(a > 0 || (a == 0 && b != 0));
    return std::exp(b * std::log(a));
}

#ifdef __CUDACC__
CELER_WRAP_MATH_FLOAT_DBL_1(, rsqrt)
#else
//---------------------------------------------------------------------------//
/*!
 * Calculate an inverse square root.
 */
inline CELER_FUNCTION double rsqrt(double value)
{
    return 1.0 / std::sqrt(value);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate an inverse square root.
 */
inline CELER_FUNCTION float rsqrt(float value)
{
    return 1.0f / std::sqrt(value);
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Integer division, rounding up, for positive numbers.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T ceil_div(T top, T bottom)
{
    static_assert(std::is_unsigned<T>::value, "Value is not an unsigned int");
    return (top / bottom) + (top % bottom != 0);
}

//---------------------------------------------------------------------------//
/*!
 * Negation that won't return signed zeros.
 */
template<class T>
[[nodiscard]] CELER_CONSTEXPR_FUNCTION T negate(T value)
{
    return T{0} - value;
}

//---------------------------------------------------------------------------//
/*!
 * Math constants (POSIX derivative);
 */
inline constexpr double m_pi = detail::m_pi;

//---------------------------------------------------------------------------//
//!@{
//! CUDA/HIP equivalent routines
#if CELER_DEVICE_SOURCE
// CUDA and HIP define sinpi and sinpif, and sincospi, sincosf
CELER_WRAP_MATH_FLOAT_DBL_1(, sinpi)
CELER_WRAP_MATH_FLOAT_DBL_1(, cospi)
CELER_WRAP_MATH_FLOAT_DBL_PTR_2(, sincospi)
CELER_WRAP_MATH_FLOAT_DBL_PTR_2(, sincos)
#elif __APPLE__
// Apple defines __sinpi, __sinpif, __sincospi, ...
CELER_WRAP_MATH_FLOAT_DBL_1(__, sinpi)
CELER_WRAP_MATH_FLOAT_DBL_1(__, cospi)
CELER_WRAP_MATH_FLOAT_DBL_PTR_2(__, sincospi)
CELER_WRAP_MATH_FLOAT_DBL_PTR_2(__, sincos)
#else
using ::celeritas::detail::cospi;
using ::celeritas::detail::sinpi;
CELER_FORCEINLINE void sincos(float a, float* s, float* c)
{
    return detail::sincos(a, s, c);
}
CELER_FORCEINLINE void sincos(double a, double* s, double* c)
{
    return detail::sincos(a, s, c);
}
CELER_FORCEINLINE void sincospi(float a, float* s, float* c)
{
    return detail::sincospi(a, s, c);
}
CELER_FORCEINLINE void sincospi(double a, double* s, double* c)
{
    return detail::sincospi(a, s, c);
}
#endif
//!@}

//---------------------------------------------------------------------------//
}  // namespace celeritas
