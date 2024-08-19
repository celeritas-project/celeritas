//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 TT-Battelle, LLC, and other Celeritas developers.
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
inline void sincospi_impl(T x, T* sptr, T* cptr)
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
        x *= static_cast<T>(m_pi);
        cval = std::cos(x);
        sval = std::sin(x);
    }
    *cptr = cval;
    *sptr = sval;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of sin(x * pi)
template<class T>
inline T sinpi_impl(T v)
{
    T result;
    T unused;
    sincospi_impl(v, &result, &unused);
    return result;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of cos(x * pi)
template<class T>
inline T cospi_impl(T v)
{
    T unused;
    T result;
    sincospi_impl(v, &unused, &result);
    return result;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of sincos
template<class T>
inline void sincos_impl(T x, T* sptr, T* cptr)
{
    *sptr = std::sin(x);
    *cptr = std::cos(x);
}

//---------------------------------------------------------------------------//

CELER_FORCEINLINE void sincospif(float x, float* sptr, float* cptr)
{
    return sincospi_impl(x, sptr, cptr);
}

CELER_FORCEINLINE void sincosf(float x, float* sptr, float* cptr)
{
    return sincos_impl(x, sptr, cptr);
}

CELER_FORCEINLINE float sinpif(float x)
{
    return sinpi_impl(x);
}

CELER_FORCEINLINE float cospif(float x)
{
    return cospi_impl(x);
}

CELER_FORCEINLINE void sincospi(double x, double* sptr, double* cptr)
{
    return sincospi_impl(x, sptr, cptr);
}

CELER_FORCEINLINE void sincos(double x, double* sptr, double* cptr)
{
    return sincos_impl(x, sptr, cptr);
}

CELER_FORCEINLINE double sinpi(double x)
{
    return sinpi_impl(x);
}

CELER_FORCEINLINE double cospi(double x)
{
    return cospi_impl(x);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
