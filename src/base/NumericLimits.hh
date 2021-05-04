//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NumericLimits.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cfloat>
#include <climits>
#ifdef __CUDA_ARCH__
#    include <math_constants.h>
#else
#    include <limits>
#endif
#include "Macros.hh"

namespace celeritas
{
/*!
 */
template<class Numeric>
struct numeric_limits;

template<>
struct numeric_limits<float>
{
    static CELER_CONSTEXPR_FUNCTION float epsilon() { return FLT_EPSILON; }
    static CELER_CONSTEXPR_FUNCTION float max() { return FLT_MAX; }

#ifndef __CUDA_ARCH__
    static float quiet_NaN()
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    static float infinity() { return std::numeric_limits<float>::infinity(); }
#else
    static CELER_FUNCTION float quiet_NaN() { return CUDART_NAN_F; }
    static CELER_FUNCTION float infinity() { return CUDART_INF_F; }
#endif
};

template<>
struct numeric_limits<double>
{
    static CELER_CONSTEXPR_FUNCTION double epsilon() { return DBL_EPSILON; }
    static CELER_CONSTEXPR_FUNCTION double max() { return DBL_MAX; }

#ifndef __CUDA_ARCH__
    static double quiet_NaN()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    static double infinity()
    {
        return std::numeric_limits<double>::infinity();
    }
#else
    static CELER_FUNCTION double quiet_NaN() { return CUDART_NAN; }
    static CELER_FUNCTION double infinity() { return CUDART_INF; }
#endif
};

template<>
struct numeric_limits<unsigned int>
{
    static CELER_CONSTEXPR_FUNCTION unsigned int max() { return UINT_MAX; }
};

template<>
struct numeric_limits<unsigned long>
{
    static CELER_CONSTEXPR_FUNCTION unsigned long max() { return ULONG_MAX; }
};

template<>
struct numeric_limits<unsigned long long>
{
    static CELER_CONSTEXPR_FUNCTION unsigned long long max()
    {
        return ULLONG_MAX;
    }
};

} // namespace celeritas
