//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Constants.hh
//---------------------------------------------------------------------------//
#ifndef base_Constants_hh
#define base_Constants_hh

#ifdef __CUDA_ARCH__
#    include <math_constants.h>
#else
#    include <limits>
#endif

namespace celeritas
{
#ifdef __CUDA_ARCH__
template<typename Numeric>
struct numeric_limits;

template<>
struct numeric_limits<float>
{
    static constexpr __device__ float epsilon() { return FLT_EPSILON; }
    static constexpr __device__ float quiet_NaN() { return CUDART_NAN_F; }
    static constexpr __device__ float infinity() { return CUDART_INF_F; }
};

template<>
struct numeric_limits<double>
{
    static constexpr __device__ double epsilon() { return DBL_EPSILON; }
    static constexpr __device__ double quiet_NaN() { return CUDART_NAN; }
    static constexpr __device__ double infinity() { return CUDART_INF; }
};

#else // not __CUDA_ARCH__

// Use default numeric limits
template<class Numeric>
using numeric_limits = std::numeric_limits<Numeric>;

#endif // __CUDA_ARCH__
//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // base_Constants_hh
