//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#    define CELER_UNLIKELY(COND) (COND)
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
/*!
 * \def CELER_ASSUME
 *
 * Add an always-on compiler assumption about the input data. This should
 * be used very rarely, and perhaps in addition to a CELER_EXPECT macro that
 * makes a similar assertion. Sometimes informing the compiler of an assumption
 * (such as the maximum range of an integer variable) can reduce code bloat and
 * silence odd warnings.
 */
#if (!defined(__CUDA_ARCH__) && (defined(__clang__) || defined(__GNUC__))) \
    || defined(__NVCOMPILER)                                               \
    || (defined(__CUDA_ARCH__) && CUDART_VERSION >= 11030)                 \
    || defined(__HIP_DEVICE_COMPILE__)
#    define CELER_UNREACHABLE __builtin_unreachable()
#    define CELER_ASSUME(COND)                 \
        do                                     \
        {                                      \
            if (__builtin_expect(!!(COND), 0)) \
            {                                  \
                __builtin_unreachable();       \
            }                                  \
        } while (0)
#elif defined(_MSC_VER)
#    define CELER_UNREACHABLE __assume(false)
#    define CELER_ASSUME(COND) __assume(COND)
#else
#    define CELER_UNREACHABLE
#    define CELER_ASSUME(COND) (void)sizeof(COND)
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

/*!
 * \def CELER_TRY_HANDLE
 *
 * "Try" to execute the statement, and "handle" *all* thrown errors by calling
 * the given function-like error handler with a \c std::exception_ptr object.
 *
 * \note A file that uses this macro must include the \c \<exception\> header
 * (but since the \c HANDLE_EXCEPTION needs to take an exception pointer, it's
 * got to be included anyway).
 */
#define CELER_TRY_HANDLE(STATEMENT, HANDLE_EXCEPTION)   \
    do                                                  \
    {                                                   \
        try                                             \
        {                                               \
            STATEMENT;                                  \
        }                                               \
        catch (...)                                     \
        {                                               \
            HANDLE_EXCEPTION(std::current_exception()); \
        }                                               \
    } while (0)

/*!
 * \def CELER_TRY_HANDLE_CONTEXT
 *
 * Try the given statement, and if it fails, chain it into the given exception.
 *
 * The given \c CONTEXT_EXCEPTION must be an expression that yields an rvalue
 * to a \c std::exception subclass that isn't \c final . The resulting chained
 * exception will be passed into \c HANDLE_EXCEPTION for processing.
 */
#define CELER_TRY_HANDLE_CONTEXT(                          \
    STATEMENT, HANDLE_EXCEPTION, CONTEXT_EXCEPTION)        \
    CELER_TRY_HANDLE(                                      \
        do {                                               \
            try                                            \
            {                                              \
                STATEMENT;                                 \
            }                                              \
            catch (...)                                    \
            {                                              \
                std::throw_with_nested(CONTEXT_EXCEPTION); \
            }                                              \
        } while (0),                                       \
        HANDLE_EXCEPTION)

/*!
 * \def CELER_DEFAULT_COPY_MOVE
 *
 * Explicitly declares defaulted copy and move constructors ans assignment
 * operators.
 */
#define CELER_DEFAULT_COPY_MOVE(CLS)      \
    CLS(CLS const&) = default;            \
    CLS& operator=(CLS const&) = default; \
    CLS(CLS&&) = default;                 \
    CLS& operator=(CLS&&) = default;

/*!
 * \def CELER_DELETE_COPY_MOVE
 *
 * Explicitly declares deleted copy and move constructors ans assignment
 * operators.
 */
#define CELER_DELETE_COPY_MOVE(CLS)      \
    CLS(CLS const&) = delete;            \
    CLS& operator=(CLS const&) = delete; \
    CLS(CLS&&) = delete;                 \
    CLS& operator=(CLS&&) = delete;
