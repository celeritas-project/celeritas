//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Macros.hh
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
/*!
 * \def CELER_FORCEINLINE_FUNCTION
 *
 * Like CELER_FUNCTION but forces inlining. Compiler optimizers usually can
 * tell what needs optimizing, but this function can provide speedups (and
 * smaller sampling profiles) when inlining optimizations are not enabled. It
 * should be used sparingly.
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
 * \def CELER_SHIELD_DEVICE
 *
 * True/false macro definition for hiding host-code-only includes from the
 * __device__ build phase. This macro can substantially improve NVCC build
 * times by preventing the compiler from having to read and write large chunks
 * of the standard library to the .cpp1.ii device compilation phase. However,
 * enabling the option will prevent the use of management classes such as
 * CollectionBuilder from working in .cu files.
 */
#if defined(__CUDA_ARCH__) && CELERITAS_SHIELD_DEVICE
#    define CELER_SHIELD_DEVICE 1
#else
#    define CELER_SHIELD_DEVICE 0
#endif

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
 * especially useful for debug-only variables and "celeritas::range" loop
 * variables where the index is left unused.
 *
 * \code
   for (CELER_MAYBE_UNUSED int x : range(100))
   {
       do_noop();
   }
   \endcode
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
 * or https://msdn.microsoft.com/en-us/library/1b3fsfxw.aspx=
 *
 * \note This macro should not generally be used; instead, the macro \c
 * CELER_ASSERT_UNREACHABLE() defined in base/Assert.hh should be used instead
 * (to provide a more detailed error message in case the point *is* reached).
 */
#if defined(__clang__) || defined(__GNUC__)
#    define CELER_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#    define CELER_UNREACHABLE __assume(false)
#else
#    define CELER_UNREACHABLE
#endif
