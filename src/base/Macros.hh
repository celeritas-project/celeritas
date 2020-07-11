//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Macros.hh
//---------------------------------------------------------------------------//
#ifndef base_Macros_hh
#define base_Macros_hh

/*!
 * \def CELER_FUNCTION
 *
 * Decorate a function that works on both host and device, with and without
 * NVCC. This is meant for small utility classes (trivially copyable!) that
 * will be used on host and device, not for complicated classes.
 */
#if defined(__NVCC__)
#    define CELER_FUNCTION __host__ __device__
#    define CELER_FORCEINLINE_FUNCTION __host__ __device__ __forceinline__
#else
#    define CELER_FUNCTION
#    if defined(_MSC_VER)
#        define CELER_FORCEINLINE_FUNCTION inline __forceinline
#    elif defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)
#        define CELER_FORCEINLINE_FUNCTION \
            inline __attribute__((always_inline))
#    else
#        define CELER_FORCEINLINE_FUNCTION inline
#    endif
#endif

/*!
 * \def CELER_CONSTEXPR_FUNCTION
 *
 * Decorate a function that works on both host and device, with and without
 * NVCC, can be evaluated at compile time, and should be forcibly inlined.
 */
#define CELER_CONSTEXPR_FUNCTION constexpr CELER_FORCEINLINE_FUNCTION

/*!
 * \def CELER_UNLIKELY(condition)
 *
 * Mark the result of this condition to be "unlikely".
 *
 * This asks the compiler to move the section of code to a "cold" part of the
 * instructions, improving instruction locality. It should be used primarily
 * for error checking conditions.
 */

#if defined(__clang__) || defined(__GNUC__)
// GCC and Clang support the same builtin
#    define CELER_UNLIKELY(COND) __builtin_expect(!!(COND), 0)
#else
// No other compilers seem to have a similar builtin
#    define CELER_UNLIKELY(COND) COND
#endif

#endif // base_Macros_hh
