//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file corecel/Macros.hh
 * \brief Language and compiler abstraction macro definitions.
 *
 * The Macros file defines cross-platform (CUDA, C++, HIP) macros that
 * expand to attributes depending on the compiler and build configuration.
 */
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

/*!
 * \def CELER_FUNCTION
 *
 * Decorate a function that works on both host and device, with and without
 * NVCC. The name of this function and its siblings is based on the Kokkos
 * naming scheme.
 */
#if defined(__NVCC__) || defined(__HIP__)
#    define CELER_FUNCTION __host__ __device__
#else
#    define CELER_FUNCTION
#endif

#if defined(__NVCC__)
#    define CELER_FORCEINLINE __forceinline__
#elif defined(_MSC_VER)
#    define CELER_FORCEINLINE inline __forceinline
#elif defined(__clang__) || defined(__GNUC__) || defined(__HIP__) \
    || defined(__INTEL_COMPILER)
#    define CELER_FORCEINLINE inline __attribute__((always_inline))
#else
#    define CELER_FORCEINLINE inline
#endif

/*!
 * \def CELER_FORCEINLINE_FUNCTION
 *
 * Like CELER_FUNCTION but forces inlining. Compiler optimizers usually can
 * tell what needs optimizing, but this function can provide speedups (and
 * smaller sampling profiles) when inlining optimizations are not enabled. It
 * should be used sparingly.
 */
#define CELER_FORCEINLINE_FUNCTION CELER_FUNCTION CELER_FORCEINLINE

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

/*!
 * \def CELER_MAYBE_UNUSED
 *
 * Mark a function, type, or variable as being potentially unused. This is
 * especially useful for debug-only variables and \c celeritas::range loop
 * variables where the index is left unused.
 *
 * \code
   for (CELER_MAYBE_UNUSED int x : range(100))
   {
       do_noop();
   }
 * \endcode
 */
#if __cplusplus >= 201710L
#    define CELER_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__)
// Valid for GCC 4.8+ and Clang
#    define CELER_MAYBE_UNUSED [[gnu::unused]]
#else
#    define CELER_MAYBE_UNUSED
#endif

/*!
 * \def CELER_UNREACHABLE
 *
 * Mark a point in code as being impossible to reach in normal execution.
 *
 * See https://clang.llvm.org/docs/LanguageExtensions.html#builtin-unreachable
 * or https://msdn.microsoft.com/en-us/library/1b3fsfxw.aspx
 * or
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#__builtin_unreachable
 *
 * (The "unreachable" and "assume" compiler optimizations for CUDA are only
 * available in API version 11.3 or higher, which is encoded as
 * \code major*1000 + minor*10 \endcode).
 *
 * \note This macro should not generally be used; instead, the macro \c
 * CELER_ASSERT_UNREACHABLE() defined in base/Assert.hh should be used instead
 * (to provide a more detailed error message in case the point *is* reached).
 */
#if (!defined(__CUDA_ARCH__) && (defined(__clang__) || defined(__GNUC__))) \
    || defined(__NVCOMPILER)                                               \
    || (defined(__CUDA_ARCH__) && CUDART_VERSION >= 11030)                 \
    || defined(__HIP_DEVICE_COMPILE__)
#    define CELER_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#    define CELER_UNREACHABLE __assume(false)
#else
#    define CELER_UNREACHABLE
#endif

/*!
 * \def CELER_USE_DEVICE
 *
 * True if HIP or CUDA are enabled, false otherwise.
 */
#if CELERITAS_USE_CUDA || CELERITAS_USE_HIP
#    define CELER_USE_DEVICE 1
#else
#    define CELER_USE_DEVICE 0
#endif

/*!
 * \def CELER_DEVICE_SOURCE
 *
 * Defined and true if building a HIP or CUDA source file. This is a generic
 * replacement for \c __CUDACC__ .
 */
#if defined(__CUDACC__) || defined(__HIP__)
#    define CELER_DEVICE_SOURCE 1
#elif defined(__DOXYGEN__)
#    define CELER_DEVICE_SOURCE 0
#endif

/*!
 * \def CELER_DEVICE_COMPILE
 *
 * Defined and true if building device code in HIP or CUDA. This is a generic
 * replacement for \c __CUDA_ARCH__ .
 */
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#    define CELER_DEVICE_COMPILE 1
#elif defined(__DOXYGEN__)
#    define CELER_DEVICE_COMPILE 0
#endif

/*!
 * \def CELER_DEVICE_PREFIX
 *
 * Add a prefix "hip" or "cuda" to a code token.
 */
#if CELERITAS_USE_CUDA
#    define CELER_DEVICE_PREFIX(TOK) cuda##TOK
#elif CELERITAS_USE_HIP
#    define CELER_DEVICE_PREFIX(TOK) hip##TOK
#else
#    define CELER_DEVICE_PREFIX(TOK) DEVICE_UNAVAILABLE
#endif
