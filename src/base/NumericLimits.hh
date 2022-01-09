//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NumericLimits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cfloat>
#include <climits>
#include "Macros.hh"

namespace celeritas
{
#define SCCEF_ static CELER_CONSTEXPR_FUNCTION

//---------------------------------------------------------------------------//
/*!
 * Subset of numeric limits compatible with both host and device.
 *
 * \note CUDART_NAN* and \c CUDART_INF* are not constexpr in CUDA 10 at least,
 *   so we have replaced those with compiler built-ins that work in GCC, Clang,
 *   and MSVC.
 */
template<class Numeric>
struct numeric_limits;

template<>
struct numeric_limits<float>
{
    SCCEF_ float epsilon() { return FLT_EPSILON; }
    SCCEF_ float max() { return FLT_MAX; }
    SCCEF_ float quiet_NaN() { return __builtin_nanf(""); }
    SCCEF_ float infinity() { return __builtin_huge_valf(); }
};

template<>
struct numeric_limits<double>
{
    SCCEF_ double epsilon() { return DBL_EPSILON; }
    SCCEF_ double max() { return DBL_MAX; }
    SCCEF_ double quiet_NaN() { return __builtin_nan(""); }
    SCCEF_ double infinity() { return __builtin_huge_val(); }
};

template<>
struct numeric_limits<unsigned int>
{
    SCCEF_ unsigned int max() { return UINT_MAX; }
};

template<>
struct numeric_limits<unsigned long>
{
    SCCEF_ unsigned long max() { return ULONG_MAX; }
};

template<>
struct numeric_limits<unsigned long long>
{
    SCCEF_ unsigned long long max() { return ULLONG_MAX; }
};

#undef SCCEF_
} // namespace celeritas
