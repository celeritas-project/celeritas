//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "corecel/Config.hh"

//---------------------------------------------------------------------------//
//!@{
//! \name Compiler type/version macros

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

//! Detection for the current compiler isn't supported yet
#define CELER_COMPILER_UNKNOWN 0
//! Compiling with clang, or a clang-based compiler defining __clang__ (hipcc)
#define CELER_COMPILER_CLANG 1
/*!
 * \def CELER_COMPILER
 *
 * Compare to on of the CELER_COMPILER_<compiler> macros to check
 * which compiler is in use.
 *
 * TODO: add and test more compilers as needed.
 */
#if defined(__clang__)
#    define CELER_COMPILER CELER_COMPILER_CLANG
#else
#    define CELER_COMPILER CELER_COMPILER_UNKNOWN
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

// For use in detail code only for thin wrapper functions
#define CELER_FIF CELER_FORCEINLINE_FUNCTION

/*!
 * \def CELER_CONSTEXPR_FUNCTION
 *
 * Decorate a function that works on both host and device, with and without
 * NVCC, can be evaluated at compile time, and should be forcibly inlined.
 */
#define CELER_CONSTEXPR_FUNCTION constexpr CELER_FORCEINLINE_FUNCTION

// For use in detail code only for thin wrapper functions
#define CELER_CEF CELER_CONSTEXPR_FUNCTION

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
 * \def CELER_USE_THRUST
 * \def CELER_CUB_HAS_TRANSFORM
 * \def CELER_CUB_HAS_FLAGGEDIF
 * \def CELER_HIPCUB_HAS_TRANSFORM
 *
 * Determine if CUB or hipCUB is available, anf if so, check the version.
 *
 * CUDA has included CUB since CUDA 11, but ROCm does not include hipCUB by
 * default, so test for the availability of hipCUB and use thrust instead if
 * it's unavailable.
 *
 * DeviceTransform is unavailable in earlier versions of CUB/hipCUB, so the
 * code uses thrust::transform in that case.
 *
 * DeviceSelect::FlaggedIf is unavailable in earlier versions of CUB and
 * doesn't work with hipCUB versions 3.4.0 through 4.1.0 when using celeritas
 * OpaqueId data types, so use a transform and DeviceSelect::Flagged instead.
 *
 * \note These are unneeded and undefined if this isn't a HIP or CUDA source
 * file. This check is necessary because the version header file needs to be
 * included.
 */
#if CELER_DEVICE_SOURCE
#    if CELERITAS_USE_CUDA
#        include <cub/version.cuh>
#    elif CELERITAS_USE_HIP && CELERITAS_HAVE_HIPCUB
#        include <hipcub/hipcub_version.hpp>
#    endif
#    if CELERITAS_USE_HIP && !CELERITAS_HAVE_HIPCUB
#        define CELER_USE_THRUST 1
#    endif
#    if CELERITAS_USE_CUDA && CUB_VERSION >= 200800
#        define CELER_CUB_HAS_TRANSFORM 1
#        define CELER_CUB_HAS_FLAGGEDIF 1
#    elif CELERITAS_USE_CUDA && CUB_VERSION >= 200500
#        define CELER_CUB_HAS_FLAGGEDIF 1
#    elif CELERITAS_USE_HIP && HIPCUB_VERSION >= 400100
#        define CELER_HIPCUB_HAS_TRANSFORM 1
#    endif
#elif defined(__DOXYGEN__)
#    define CELER_USE_THRUST 0
#    define CELER_CUB_HAS_TRANSFORM 0
#    define CELER_CUB_HAS_FLAGGEDIF 0
#    define CELER_HIPCUB_HAS_TRANSFORM 0
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

#if CELERITAS_USE_CUDA            \
    && (__CUDACC_VER_MAJOR__ < 11 \
        || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 5))
/*!
 * Work around older NVCC bugs with `if constexpr`.
 *
 * These cause errors such as \verbatim
 *    error: missing return statement at end of non-void function
 * \endverbatim
 */
#    define CELER_CUDACC_BUGGY_IF_CONSTEXPR 1
#else
#    define CELER_CUDACC_BUGGY_IF_CONSTEXPR 0
#endif

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name Exception handling macros

/*!
 * \def CELER_TRY_HANDLE
 *
 * "Try" to execute the statement, and "handle" *all* thrown errors by calling
 * the given function-like error handler with a \c std::exception_ptr object.
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
 * to a \c std::exception subclass that is not marked \c final . The resulting
 * chained exception will be passed into \c HANDLE_EXCEPTION for processing.
 *
 * This example shows how to safely propagate exceptions out of an OpenMP
 * parallel block using celeritas::MultiExceptionHandler :
 * \code
    MultiExceptionHandler capture_exception;
    #pragma omp parallel for
    for (size_type i = 0; i < num_threads; ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            execute_thread(ThreadId{i}),
            capture_exception,
            KernelContextException(ThreadId{i}));
    }
    log_and_rethrow(std::move(capture_exception));
 * \endcode
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

//!@}
//---------------------------------------------------------------------------//
//!@{
//! \name Class definition macros

/*!
 * \def CELER_DEFAULT_COPY_MOVE
 *
 * Explicitly declare defaulted copy and move constructors and assignment
 * operators. Use this if the destructor is declared explicitly, or as part of
 * the "protected" section of an interface class to prevent assignment between
 * incompatible classes.
 */
#define CELER_DEFAULT_COPY_MOVE(CLS)      \
    CLS(CLS const&) = default;            \
    CLS& operator=(CLS const&) = default; \
    CLS(CLS&&) = default;                 \
    CLS& operator=(CLS&&) = default

/*!
 * \def CELER_DELETE_COPY_MOVE
 *
 * Explicitly declare *deleted* copy and move constructors and assignment
 * operators. Use this for scoped RAII classes.
 */
#define CELER_DELETE_COPY_MOVE(CLS)      \
    CLS(CLS const&) = delete;            \
    CLS& operator=(CLS const&) = delete; \
    CLS(CLS&&) = delete;                 \
    CLS& operator=(CLS&&) = delete

/*!
 * \def CELER_DEFAULT_MOVE_DELETE_COPY
 *
 * Explicitly declare defaulted copy and move constructors and assignment
 * operators. Use this if the destructor is declared explicitly.
 */
#define CELER_DEFAULT_MOVE_DELETE_COPY(CLS) \
    CLS(CLS const&) = delete;               \
    CLS& operator=(CLS const&) = delete;    \
    CLS(CLS&&) = default;                   \
    CLS& operator=(CLS&&) = default

//!@}
//---------------------------------------------------------------------------//

/*!
 * \def CELER_DISCARD
 *
 * The argument is an unevaluated operand which will generate no code but force
 * the expression to be used. This is used in place of the \code
 * [[maybe_unused]] \endcode attribute, which actually generates warnings in
 * older versions of GCC.
 */
#define CELER_DISCARD(CODE) static_cast<void>(sizeof(CODE));

//---------------------------------------------------------------------------//
