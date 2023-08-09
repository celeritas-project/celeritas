//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 TT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/MathImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
inline constexpr double m_pi{3.14159265358979323846};

//---------------------------------------------------------------------------//
/*!
 * Calculate sin and cosine of a fraction of pi.
 *
 * This returns exact values for half-fractions of pi and
 * results in more accurate values for large integers.
 *
 * This lazy naive implementation doesn't account for edge cases like NaN and
 * signed zeros, and it's not going to be efficient.
 */
template<class T>
inline void sincospi(T x, T* sptr, T* cptr)
{
    // Note: fmod returns value in (-2, 2)
    x = std::fmod(x, T{2}) + (x < 0 ? T{2} : 0);
    T cval;
    T sval;
    if (x == T{0})
    {
        cval = 1;
        sval = 0;
    }
    else if (x == T{1} / T{2})
    {
        cval = 0;
        sval = 1;
    }
    else if (x == T{1})
    {
        cval = -1;
        sval = 0;
    }
    else if (x == T{3} / T{2})
    {
        cval = 0;
        sval = -1;
    }
    else
    {
        x *= m_pi;
        cval = std::cos(x);
        sval = std::sin(x);
    }
    *cptr = cval;
    *sptr = sval;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of sin(x * pi)
template<class T>
inline T sinpi(T v)
{
    T result;
    T unused;
    sincospi(v, &result, &unused);
    return result;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of cos(x * pi)
template<class T>
inline T cospi(T v)
{
    T unused;
    T result;
    sincospi(v, &unused, &result);
    return result;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of sincos
template<class T>
inline void sincos(T x, T* sptr, T* cptr)
{
    *sptr = std::sin(x);
    *cptr = std::cos(x);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
