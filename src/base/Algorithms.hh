//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Algorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return the lower of two values.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION const T& min(const T& a, const T& b)
{
    return (b < a) ? b : a;
}

//---------------------------------------------------------------------------//
/*!
 * Return the higher of two values.
 */
template<class T>
CELER_CONSTEXPR_FUNCTION const T& max(const T& a, const T& b)
{
    return (b > a) ? b : a;
}

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
CELER_CONSTEXPR_FUNCTION T ipow(T v)
{
    return (N == 0)       ? 1
           : (N % 2 == 0) ? ipow<N / 2>(v) * ipow<N / 2>(v)
                          : v * ipow<(N - 1) / 2>(v) * ipow<(N - 1) / 2>(v);
}

//---------------------------------------------------------------------------//
/*!
 * Interchange (swap) the values of two inputs.
 */
template<typename T>
CELER_CONSTEXPR_FUNCTION void swap2(T& v1, T& v2)
{
    T tmp = v1;
    v1    = v2;
    v2    = tmp;
}

//---------------------------------------------------------------------------//
/*!
 * Return the cube of the input value.
 */
template<class T>
[[deprecated(
    "Replace with celeritas::ipow<3>(value)")]] CELER_CONSTEXPR_FUNCTION T
cube(const T& a)
{
    return a * a * a;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
