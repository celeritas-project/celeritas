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
 * \todo Add a template on comparator if needed.
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
 * Return the lower of two values.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION const T& min(const T& a, const T& b) noexcept
{
    return (b < a) ? b : a;
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
template<class ForwardIt>
inline CELER_FUNCTION ForwardIt min_element(ForwardIt iter, ForwardIt last)
{
    // Avoid incrementing past the end
    if (iter == last)
        return last;

    ForwardIt result = iter++;
    for (; iter != last; ++iter)
    {
        if (*iter < *result)
            result = iter;
    }
    return result;
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
// Replace/extend <utility>
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
} // namespace celeritas
