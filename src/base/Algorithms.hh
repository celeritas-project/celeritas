//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Algorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include "Macros.hh"

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
        return forward<T>(lhs) < forward<U>(rhs);
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
 *
 * \todo Define an iterator adapter that dereferences using `__ldg` in
 * device code.
 * \todo Add a template on comparator if needed (defaulting to Less).
 * \todo Add a "lower_bound_index" that will use the native pointer difference
 * type instead of iterator arithmetic, for potential speedup on CUDA. Or
 * define an iterator adapter to Collections.
 */
template<class ForwardIt, class T>
inline CELER_FUNCTION ForwardIt lower_bound(ForwardIt first,
                                            ForwardIt last,
                                            const T&  value) noexcept
{
    auto count = last - first;
    while (count > 0)
    {
        auto      step   = count / 2;
        ForwardIt middle = first + step;
        if (*middle < value)
        {
            first = middle + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
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
CELER_FORCEINLINE_FUNCTION ForwardIt min_element(ForwardIt iter, ForwardIt last)
{
    return ::celeritas::min_element(iter, last, Less<decltype(*iter)>{});
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
