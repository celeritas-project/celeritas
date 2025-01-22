//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Algorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <type_traits>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

#include "NumericLimits.hh"

#include "detail/AlgorithmsImpl.hh"

#if !defined(CELER_DEVICE_SOURCE) && !defined(CELERITAS_SINCOSPI_PREFIX)
#    include "detail/Sincospi.hh"
#endif

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

//! \cond (CELERITAS_DOC_DEV)
template<class T>
CELER_CONSTEXPR_FUNCTION T&&
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
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
 * Replace a value and return the original.
 *
 * This has a similar signature to atomic updates.
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
    CELER_CONSTEXPR_FUNCTION auto
    operator()(T&& lhs, U&& rhs) const -> decltype(auto)
    {
        return ::celeritas::forward<T>(lhs) < ::celeritas::forward<U>(rhs);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Evaluate whether the argument is "true".
 *
 * This is useful for calls to \c std::all_of .
 */
template<class T = void>
struct LogicalTrue
{
    CELER_CONSTEXPR_FUNCTION bool operator()(T const& value) const noexcept
    {
        return static_cast<bool>(value);
    }
};

//! Specialization of LogicalTrue with template deduction
template<>
struct LogicalTrue<void>
{
    template<class T>
    CELER_CONSTEXPR_FUNCTION bool operator()(T const& value) const noexcept
    {
        return static_cast<bool>(value);
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
 * Whether the predicate is true for pairs of consecutive items.
 */
template<class InputIt, class Predicate>
inline CELER_FUNCTION bool all_adjacent(InputIt iter, InputIt last, Predicate p)
{
    if (iter == last)
        return true;

    auto prev = *iter++;
    while (iter != last)
    {
        if (!p(prev, *iter))
            return false;
        prev = *iter++;
    }
    return true;
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
    return (v < T{0}) ? T{0} : v;
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

//! \cond (CELERITAS_DOC_DEV)
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
//! \endcond

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

//! \cond (CELERITAS_DOC_DEV)
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
//! \endcond

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

//! \cond (CELERITAS_DOC_DEV)
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
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Find the given element in a sorted range.
 */
template<class ForwardIt, class T, class Compare>
inline CELER_FUNCTION ForwardIt
find_sorted(ForwardIt first, ForwardIt last, T const& value, Compare comp)
{
    auto iter = ::celeritas::lower_bound(first, last, value, comp);
    if (iter == last || comp(*iter, value) || comp(value, *iter))
    {
        // Insertion point is off the end, or value is not equal
        return last;
    }
    return iter;
}

//!\cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
/*!
 * Find the given element in a sorted range.
 */
template<class ForwardIt, class T>
CELER_FORCEINLINE_FUNCTION ForwardIt find_sorted(ForwardIt first,
                                                 ForwardIt last,
                                                 T const& value)
{
    return ::celeritas::find_sorted(first, last, value, Less<>{});
}
//!\endcond

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

//! \cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
/*!
 * Sort an array on a single thread.
 */
template<class RandomAccessIt>
CELER_FORCEINLINE_FUNCTION void sort(RandomAccessIt first, RandomAccessIt last)
{
    ::celeritas::sort(first, last, Less<>{});
}
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Return the higher of two values.
 *
 * This function is specialized so that floating point types use \c std::fmax
 * for better performance on GPU and ARM.
 */
template<class T, std::enable_if_t<!std::is_floating_point<T>::value, bool> = true>
CELER_CONSTEXPR_FUNCTION T const& max(T const& a, T const& b) noexcept
{
    return (b > a) ? b : a;
}

//!\cond (CELERITAS_DOC_DEV)
template<class T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
CELER_CONSTEXPR_FUNCTION T max(T a, T b) noexcept
{
    return std::fmax(a, b);
}
//!\endcond

//---------------------------------------------------------------------------//
/*!
 * Return the lower of two values.
 *
 * This function is specialized so that floating point types use \c std::fmin
 * for better performance on GPU and ARM.
 */
template<class T, std::enable_if_t<!std::is_floating_point<T>::value, bool> = true>
CELER_CONSTEXPR_FUNCTION T const& min(T const& a, T const& b) noexcept
{
    return (b < a) ? b : a;
}

//!\cond (CELERITAS_DOC_DEV)
template<class T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
CELER_CONSTEXPR_FUNCTION T min(T a, T b) noexcept
{
    return std::fmin(a, b);
}
//!\endcond

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

//!\cond (CELERITAS_DOC_DEV)
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
//!\endcond

//---------------------------------------------------------------------------//
// Replace/extend <cmath>
//---------------------------------------------------------------------------//
/*!
 * Return a nonnegative integer power of the input value.
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
        CELER_DISCARD(v)  // Suppress warning in older compilers
        return T{1};
    }
    else if constexpr (N % 2 == 0)
    {
        return ipow<N / 2>(v) * ipow<N / 2>(v);
    }
    else
    {
        return v * ipow<(N - 1) / 2>(v) * ipow<(N - 1) / 2>(v);
    }
#if (__CUDACC_VER_MAJOR__ < 11) \
    || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 5)
    // "error: missing return statement at end of non-void function"
    return T{0};
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Raise a number to a power with simplifying assumptions.
 *
 * This should be faster than \c std::pow because we don't worry about
 * special cases for zeros, infinities, or negative values for \c a.
 *
 * Example: \code
  assert(9.0 == fastpow(3.0, 2.0));
 \endcode
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
inline CELER_FUNCTION T fastpow(T a, T b)
{
    CELER_EXPECT(a > 0 || (a == 0 && b != 0));
    return std::exp(b * std::log(a));
}

//---------------------------------------------------------------------------//
/*!
 * Use fused multiply-add for generic calculations.
 *
 * This provides a floating point specialization so that \c fma can be used in
 * code that is accelerated for floating point calculations but still works
 * correctly with integer arithmetic.
 *
 * Because of the single template parameter, it may be easier to use \c
 * std::fma directly in most cases.
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
CELER_FORCEINLINE_FUNCTION T fma(T a, T b, T y)
{
    return std::fma(a, b, y);
}

//---------------------------------------------------------------------------//
/*!
 * Provide an FMA-like interface for integers.
 */
template<class T, std::enable_if_t<!std::is_floating_point<T>::value, bool> = true>
CELER_CONSTEXPR_FUNCTION T fma(T a, T b, T y)
{
    return a * b + y;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate a hypotenuse.
 *
 * This does \em not conform to IEEE754 by returning infinity in edge cases
 * (e.g., one argument is infinite and the other NaN). Similarly, it is not
 * symmetric with respect to the function arguments.
 *
 * To improve accuracy we could use [1].
 *
 * [1] C.F. Borges, An Improved Algorithm for hypot(a,b), (2019).
 *     http://arxiv.org/abs/1904.09481 (accessed November 19, 2024).
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T hypot(T a, T b)
{
    return std::sqrt(fma(b, b, ipow<2>(a)));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate a hypotenuse.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T hypot(T a, T b, T c)
{
    T result = fma(b, b, ipow<2>(a));
    result = fma(c, c, result);
    return std::sqrt(result);
}

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
 * Calculate local work for a given worker ID.
 *
 * This calculates the amount of local work for the given worker ID when
 * dividing \c total_work tasks over \c num_workers workers.
 */
template<class T>
struct LocalWorkCalculator
{
    static_assert(std::is_unsigned<T>::value, "Value is not an unsigned int");

    T total_work;
    T num_workers;

    CELER_CONSTEXPR_FUNCTION T operator()(T local_id)
    {
        CELER_EXPECT(local_id < num_workers);
        return total_work / num_workers + (local_id < total_work % num_workers);
    }
};

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
 * Calculate the difference of squares \f$ a^2 - b^2 \f$.
 *
 * This calculation exchanges one multiplication for one addition, but it does
 * not increase the accuracy of the computed result. It is used
 * occasionally in Geant4 but is likely a premature optimization... see
 * https://github.com/celeritas-project/celeritas/pull/1082
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T diffsq(T a, T b)
{
    return (a - b) * (a + b);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the Euclidian modulus of two numbers.
 *
 * If both numbers are positive, this should be the same as fmod. If the
 * sign of the remainder and denominator don't match, the remainder will be
 * remapped so that it is between zero and the denominator.
 *
 * This function is useful for normalizing user-provided angles. Examples:
 * \code
   eumod(3, 2) == 1
   eumod(-0.5, 2) == 1.5
   eumod(-2, 2) == 0
   \endcode
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
CELER_CONSTEXPR_FUNCTION T eumod(T numer, T denom)
{
    T r = std::fmod(numer, denom);
    if (r < 0)
    {
        if (denom >= 0)
        {
            r += denom;
        }
        else
        {
            r -= denom;
        }
    }
    return r;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the sign of a number.
 *
 * \return -1 if negative, 0 if exactly zero (or NaN), 1 if positive
 */
template<class T>
CELER_CONSTEXPR_FUNCTION int signum(T x)
{
    return (0 < x) - (x < 0);
}

//---------------------------------------------------------------------------//
//!@{
//! \name CUDA/HIP equivalent routines

/*!
 * Calculate an inverse square root.
 */
CELER_FORCEINLINE_FUNCTION float rsqrt(float value)
{
#ifdef __CUDACC__
    return ::rsqrtf(value);
#else
    // NOTE: some C++ library implementations (GCC?) don't define
    // single-precision functions in std namespace
    return 1.0f / sqrtf(value);
#endif
}

/*!
 * Calculate an inverse square root.
 */
CELER_FORCEINLINE_FUNCTION double rsqrt(double value)
{
#ifdef __CUDACC__
    return ::rsqrt(value);
#else
    return 1.0 / std::sqrt(value);
#endif
}

#if CELER_DEVICE_SOURCE
// CUDA/HIP define sinpi, cospi, sinpif, cospif, ...
#    undef CELERITAS_SINCOSPI_PREFIX
#    define CELERITAS_SINCOSPI_PREFIX
#endif

#ifdef CELERITAS_SINCOSPI_PREFIX
// Apple-supplied headers define __sinpi, __sinpif, __sincospi, ...
#    define CELER_CONCAT_IMPL(PREFIX, FUNC) PREFIX##FUNC
#    define CELER_CONCAT(PREFIX, FUNC) CELER_CONCAT_IMPL(PREFIX, FUNC)
#    define CELER_SINCOS_MANGLED(FUNC) \
        CELER_CONCAT(CELERITAS_SINCOSPI_PREFIX, FUNC)
#else
// Use implementations from detail/MathImpl.hh
#    define CELERITAS_SINCOSPI_PREFIX ::celeritas::detail::
#    define CELER_SINCOS_MANGLED(FUNC) ::celeritas::detail::FUNC
#endif

//!@{
//! Get the sine or cosine of a value multiplied by pi for increased precision
CELER_FORCEINLINE_FUNCTION float sinpi(float a)
{
    return CELER_SINCOS_MANGLED(sinpif)(a);
}
CELER_FORCEINLINE_FUNCTION double sinpi(double a)
{
    return CELER_SINCOS_MANGLED(sinpi)(a);
}
CELER_FORCEINLINE_FUNCTION float cospi(float a)
{
    return CELER_SINCOS_MANGLED(cospif)(a);
}
CELER_FORCEINLINE_FUNCTION double cospi(double a)
{
    return CELER_SINCOS_MANGLED(cospi)(a);
}
//!@}

//!@{
//! Simultaneously evaluate the sine and cosine of a value.
CELER_FORCEINLINE_FUNCTION void sincos(float a, float* s, float* c)
{
    return CELER_SINCOS_MANGLED(sincosf)(a, s, c);
}
CELER_FORCEINLINE_FUNCTION void sincos(double a, double* s, double* c)
{
    return CELER_SINCOS_MANGLED(sincos)(a, s, c);
}
//!@}

//!@{
//! Simultaneously evaluate the sine and cosine of a value factored by pi.
CELER_FORCEINLINE_FUNCTION void sincospi(float a, float* s, float* c)
{
    return CELER_SINCOS_MANGLED(sincospif)(a, s, c);
}
CELER_FORCEINLINE_FUNCTION void sincospi(double a, double* s, double* c)
{
    return CELER_SINCOS_MANGLED(sincospi)(a, s, c);
}
//!@}

//!@}

//---------------------------------------------------------------------------//
// Portable utilities functions
//---------------------------------------------------------------------------//
/*!
 * Count the number of set bits in an integer.
 */
template<class T>
#if defined(_MSC_VER)
CELER_FORCEINLINE_FUNCTION int popcount(T x) noexcept
#else
CELER_CONSTEXPR_FUNCTION int popcount(T x) noexcept
#endif
{
    static_assert(sizeof(T) <= 8,
                  "popcount is only defined for 32-bit and 64-bit integers");
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>,
                  "popcount is only defined for unsigned integral types");

    if constexpr (sizeof(T) <= 4)
    {
#if CELER_DEVICE_COMPILE
        return __popc(x);
#elif defined(_MSC_VER)
        return __popcnt(x);
#else
        return __builtin_popcount(x);
#endif
    }

#if CELER_DEVICE_COMPILE
    return __popcll(x);
#elif defined(_MSC_VER)
    return __popcnt64(x);
#else
    return __builtin_popcountl(x);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
