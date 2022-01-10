//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Algorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include "Macros.hh"
#include "detail/AlgorithmsImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Replace/extend <utility>
//---------------------------------------------------------------------------//
//!@{
//! Implement perfect forwarding with device-friendly functions.
template<class T>
CELER_CONSTEXPR_FUNCTION T&&
forward(typename std::remove_reference<T>::type& v) noexcept
{
    return static_cast<T&&>(v);
}

template<class T>
CELER_CONSTEXPR_FUNCTION T&&
forward(typename std::remove_reference<T>::type&& v) noexcept
{
    return static_cast<T&&>(v);
}
//!@}

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
// Replace/extend <functional>
//---------------------------------------------------------------------------//
/*!
 * Evaluator for the first argument being less than the second.
 */
template<class T = void>
struct Less
{
    CELER_CONSTEXPR_FUNCTION auto
    operator()(const T& lhs, const T& rhs) const noexcept -> decltype(auto)
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
 * Return the value or (if it's negative) then zero.
 *
 * This is constructed to correctly propagate NaN.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION T clamp_to_nonneg(T v) noexcept
{
    return (v < 0) ? 0 : v;
}

//---------------------------------------------------------------------------//
/*!
 * Find the insertion point for a value in a sorted list.
 */
template<class ForwardIt, class T, class Compare>
CELER_FORCEINLINE_FUNCTION ForwardIt
lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
{
    using CompareRef = std::add_lvalue_reference_t<Compare>;
    return ::celeritas::detail::lower_bound_impl<CompareRef>(
        first, last, value, comp);
}

//---------------------------------------------------------------------------//
/*!
 * Find the insertion point for a value in a sorted list.
 */
template<class ForwardIt, class T>
CELER_FORCEINLINE_FUNCTION ForwardIt lower_bound(ForwardIt first,
                                                 ForwardIt last,
                                                 const T&  value)
{
    return ::celeritas::lower_bound(first, last, value, Less<>{});
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
 */
template<class T>
CELER_CONSTEXPR_FUNCTION const T& max(const T& a, const T& b) noexcept
{
    return (b > a) ? b : a;
}

//---------------------------------------------------------------------------//
/*!
 * Return the lower of two values.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION const T& min(const T& a, const T& b) noexcept
{
    return (b < a) ? b : a;
}

//---------------------------------------------------------------------------//
/*!
 * Return an iterator to the lowest value in the range as defined by Compare.
 */
template<class ForwardIt, class Compare>
inline CELER_FUNCTION ForwardIt min_element(ForwardIt iter,
                                            ForwardIt last,
                                            Compare   comp)
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
/*!
 * Return an integer power of the input value.
 *
 * Example: \code
  assert(9.0 == ipow<2>(3.0));
  assert(256 == ipow<8>(2));
 \endcode
 */
template<unsigned int N, class T>
CELER_CONSTEXPR_FUNCTION T ipow(T v) noexcept
{
    return (N == 0)       ? 1
           : (N % 2 == 0) ? ipow<N / 2>(v) * ipow<N / 2>(v)
                          : v * ipow<(N - 1) / 2>(v) * ipow<(N - 1) / 2>(v);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
