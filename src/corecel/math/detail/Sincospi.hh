//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/Sincospi.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <cstdint>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate sin and cosine of a fraction of pi.
 * From
 * https://stackoverflow.com/questions/42792939/implementation-of-sinpi-and-cospi-using-standard-c-math-library
 * from njuffa, (c) CC BY-SA
 *
 */
inline void sincospi_impl(double a, double* sptr, double* cptr)
{
    double c, r, s, t, az;
    std::int64_t i;

    az = a * 0.0;  // must be evaluated with IEEE-754 semantics
    // for |a| >= 2**53, cospi(a) = 1.0, but cospi(Inf) = NaN
    a = (std::fabs(a) < 9.0071992547409920e+15) ? a : az;  // 0x1.0p53
    // reduce argument to primary approximation interval (-0.25, 0.25)
    r = std::nearbyint(a + a);  // must use IEEE-754 "to nearest" rounding
    i = static_cast<std::int64_t>(r);
    t = std::fma(-0.5, r, a);
    // compute core approximations
    s = t * t;
    // Approximate cos(pi*x) for x in [-0.25,0.25]
    r = -1.0369917389758117e-4;
    r = std::fma(r, s, 1.9294935641298806e-3);
    r = std::fma(r, s, -2.5806887942825395e-2);
    r = std::fma(r, s, 2.3533063028328211e-1);
    r = std::fma(r, s, -1.3352627688538006e+0);
    r = std::fma(r, s, 4.0587121264167623e+0);
    r = std::fma(r, s, -4.9348022005446790e+0);
    c = std::fma(r, s, 1.0000000000000000e+0);
    // Approximate sin(pi*x) for x in [-0.25,0.25]
    r = 4.6151442520157035e-4;
    r = std::fma(r, s, -7.3700183130883555e-3);
    r = std::fma(r, s, 8.2145868949323936e-2);
    r = std::fma(r, s, -5.9926452893214921e-1);
    r = std::fma(r, s, 2.5501640398732688e+0);
    r = std::fma(r, s, -5.1677127800499516e+0);
    s = s * t;
    r = r * s;
    s = std::fma(t, 3.1415926535897931e+0, r);
    // Map results according to quadrant
    if (i & 2)
    {
        s = 0.0 - s;  // must be evaluated with IEEE-754 semantics
        c = 0.0 - c;  // must be evaluated with IEEE-754 semantics
    }
    if (i & 1)
    {
        t = 0.0 - s;  // must be evaluated with IEEE-754 semantics
        s = c;
        c = t;
    }
    // IEEE-754: sinPi(+n) is +0 and sinPi(-n) is -0 for positive integers n
    if (a == std::floor(a))
        s = az;
    *sptr = s;
    *cptr = c;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate sin and cosine of a fraction of pi.
 * From
 * https://stackoverflow.com/questions/42792939/implementation-of-sinpi-and-cospi-using-standard-c-math-library
 * from njuffa, (c) CC BY-SA
 */
inline void sincospi_impl(float a, float* sptr, float* cptr)
{
    // NOTE: some C++ library implementations (GCC?) don't define
    // single-precision functions in std namespace
    using namespace std;

    float az, t, c, r, s;
    int32_t i;

    az = a * 0.0f;  // Must be evaluated with IEEE-754 semantics
    // For |a| > 2**24, cospi(a) = 1.0f, but cospi(Inf) = NaN
    a = (fabsf(a) < 0x1.0p24f) ? a : az;
    r = nearbyintf(a + a);  // Must use IEEE-754 "to nearest" rounding
    i = static_cast<int32_t>(r);
    t = fmaf(-0.5f, r, a);
    // Compute core approximations
    s = t * t;
    // Approximate cos(pi*x) for x in [-0.25,0.25] */
    r = 0x1.d9e000p-3f;
    r = fmaf(r, s, -0x1.55c400p+0f);
    r = fmaf(r, s, 0x1.03c1cep+2f);
    r = fmaf(r, s, -0x1.3bd3ccp+2f);
    c = fmaf(r, s, 0x1.000000p+0f);
    // Approximate sin(pi*x) for x in [-0.25,0.25] */
    r = -0x1.310000p-1f;
    r = fmaf(r, s, 0x1.46737ep+1f);
    r = fmaf(r, s, -0x1.4abbfep+2f);
    r = (t * s) * r;
    s = fmaf(t, 0x1.921fb6p+1f, r);
    if (i & 2)
    {
        s = 0.0f - s;  // Must be evaluated with IEEE-754 semantics
        c = 0.0f - c;  // Must be evaluated with IEEE-754 semantics
    }
    if (i & 1)
    {
        t = 0.0f - s;  // Must be evaluated with IEEE-754 semantics
        s = c;
        c = t;
    }
    // IEEE-754: sinPi(+n) is +0 and sinPi(-n) is -0 for positive integers n
    if (a == floorf(a))
        s = az;
    *sptr = s;
    *cptr = c;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of sin(x * pi)
template<class T>
CELER_FORCEINLINE T sinpi_impl(T v)
{
    T result;
    T unused;
    sincospi_impl(v, &result, &unused);
    return result;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of cos(x * pi)
template<class T>
CELER_FORCEINLINE T cospi_impl(T v)
{
    T unused;
    T result;
    sincospi_impl(v, &unused, &result);
    return result;
}

//---------------------------------------------------------------------------//
//! Lazy implementation of sincos
template<class T>
CELER_FORCEINLINE void sincos_impl(T x, T* sptr, T* cptr)
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
