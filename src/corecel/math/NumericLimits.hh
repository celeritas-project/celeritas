//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/NumericLimits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cfloat>
#include <climits>

#include "corecel/Macros.hh"

namespace celeritas
{
#define SCCEF_ static CELER_CONSTEXPR_FUNCTION

//---------------------------------------------------------------------------//
/*!
 * Subset of numeric limits compatible with both host and device.
 *
 * \note \c CUDART_NAN and \c CUDART_INF are not \c constexpr in CUDA 10 at
 *   least, so we have replaced those with compiler built-ins that work in GCC,
 *   Clang, and MSVC.
 *
 * \deprecated The name of this class will change to NumericLimits to conform
 * to the Celeritas naming system, since it is not a complete replacement for
 * the \c std implementation.
 */
template<class Numeric>
struct numeric_limits;

template<>
struct numeric_limits<float>
{
    SCCEF_ float epsilon() { return FLT_EPSILON; }
    SCCEF_ float lowest() { return -FLT_MAX; }
    SCCEF_ float min() { return FLT_MIN; }
    SCCEF_ float max() { return FLT_MAX; }
    SCCEF_ float quiet_NaN() { return __builtin_nanf(""); }
    SCCEF_ float infinity() { return __builtin_huge_valf(); }
};

template<>
struct numeric_limits<double>
{
    //! Relative difference between 1 and adjacent floating point number
    SCCEF_ double epsilon() { return DBL_EPSILON; }
    //! Most negative finite value
    SCCEF_ double lowest() { return -DBL_MAX; }
    //! Smallest positive value
    SCCEF_ double min() { return DBL_MIN; }
    //! Largest finite value
    SCCEF_ double max() { return DBL_MAX; }
    //! Special value for not-a-number (always compares false)
    SCCEF_ double quiet_NaN() { return __builtin_nan(""); }
    //! Special positive infinite value
    SCCEF_ double infinity() { return __builtin_huge_val(); }
};

template<>
struct numeric_limits<int>
{
    SCCEF_ int lowest() { return INT_MIN; }
    SCCEF_ int min() { return INT_MIN; }
    SCCEF_ int max() { return INT_MAX; }
};

template<>
struct numeric_limits<long>
{
    SCCEF_ long lowest() { return LONG_MIN; }
    SCCEF_ long min() { return LONG_MIN; }
    SCCEF_ long max() { return LONG_MAX; }
};

template<>
struct numeric_limits<long long>
{
    SCCEF_ long long lowest() { return LLONG_MIN; }
    SCCEF_ long long min() { return LLONG_MIN; }
    SCCEF_ long long max() { return LLONG_MAX; }
};

template<>
struct numeric_limits<unsigned int>
{
    SCCEF_ unsigned int lowest() { return 0; }
    SCCEF_ unsigned int min() { return 0; }
    SCCEF_ unsigned int max() { return UINT_MAX; }
};

template<>
struct numeric_limits<unsigned long>
{
    SCCEF_ unsigned long lowest() { return 0; }
    SCCEF_ unsigned long min() { return 0; }
    SCCEF_ unsigned long max() { return ULONG_MAX; }
};

template<>
struct numeric_limits<unsigned long long>
{
    SCCEF_ unsigned long long lowest() { return 0; }
    SCCEF_ unsigned long long min() { return 0; }
    SCCEF_ unsigned long long max() { return ULLONG_MAX; }
};

#undef SCCEF_

//! Forward-compatible alias for v1.0
template<class T>
using NumericLimits = numeric_limits<T>;

}  // namespace celeritas
