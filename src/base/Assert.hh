//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*! \file Assert.hh
 *  \brief Macros, exceptions, and helpers for assertions and error handling.
 */
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "Macros.hh"
#ifndef __CUDA_ARCH__
#    include <sstream>
#    include <stdexcept>
#    include <string>
#endif

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
/*!
 * \def CELER_EXPECT
 *
 * Precondition debug assertion macro. It is to "require" that the input values
 * or initial state satisfy a precondition.
 */
/*!
 * \def CELER_ASSERT
 *
 * Internal debug assertion macro. This replaces standard \c assert usage.
 */
/*!
 * \def CELER_ENSURE
 *
 * Postcondition debug assertion macro. Use to "ensure" that return values or
 * side effects are as expected when leaving a function.
 */
/*!
 * \def CELER_VALIDATE
 *
 * Always-on runtime assertion macro. This can check user input and input data
 * consistency, and will raise RuntimeError on failure with a descriptive error
 * message. This should not be used on device.
 */
/*!
 * \def CELER_ASSERT_UNREACHABLE
 *
 * Throw an assertion if the code point is reached. When debug assertions are
 * turned off, this changes to a compiler hint that improves optimization (and
 * may force the coded to exit uncermoniously if the point is encountered,
 * rather than continuing on with undefined behavior).
 */
/*!
 * \def CELER_NOT_CONFIGURED
 *
 * Assert if the code point is reached because an optional feature is disabled.
 * This generally should be used for the constructors of dummy class
 * definitions in, e.g., \c Foo.nocuda.cc:
 * \code
    Foo::Foo()
    {
        CELER_NOT_CONFIGURED("CUDA");
    }
 \endcode
 */

//! \cond
#define CELER_CUDA_ASSERT_(COND) \
    do                           \
    {                            \
        assert(COND);            \
    } while (0)
#define CELER_DEBUG_ASSERT_(COND, WHICH)                                        \
    do                                                                          \
    {                                                                           \
        if (CELER_UNLIKELY(!(COND)))                                            \
            ::celeritas::throw_debug_error(                                     \
                ::celeritas::DebugErrorType::WHICH, #COND, __FILE__, __LINE__); \
    } while (0)
#define CELER_DEBUG_FAIL_(MSG, WHICH)                                     \
    do                                                                    \
    {                                                                     \
        ::celeritas::throw_debug_error(                                   \
            ::celeritas::DebugErrorType::WHICH, MSG, __FILE__, __LINE__); \
    } while (0)
#define CELER_RUNTIME_ASSERT_(COND, MSG)                              \
    do                                                                \
    {                                                                 \
        if (CELER_UNLIKELY(!(COND)))                                  \
        {                                                             \
            std::ostringstream celer_runtime_msg_;                    \
            celer_runtime_msg_ << MSG;                                \
            ::celeritas::throw_runtime_error(                         \
                celer_runtime_msg_.str(), #COND, __FILE__, __LINE__); \
        }                                                             \
    } while (0)
#define CELER_NOASSERT_(COND)   \
    do                          \
    {                           \
        if (false && (COND)) {} \
    } while (0)
//! \endcond

#if CELERITAS_DEBUG && defined(__CUDA_ARCH__)
#    define CELER_EXPECT(COND) CELER_CUDA_ASSERT_(COND)
#    define CELER_ASSERT(COND) CELER_CUDA_ASSERT_(COND)
#    define CELER_ENSURE(COND) CELER_CUDA_ASSERT_(COND)
#    define CELER_ASSERT_UNREACHABLE() CELER_CUDA_ASSERT_(false)
#elif CELERITAS_DEBUG && !defined(__CUDA_ARCH__)
#    define CELER_EXPECT(COND) CELER_DEBUG_ASSERT_(COND, precondition)
#    define CELER_ASSERT(COND) CELER_DEBUG_ASSERT_(COND, internal)
#    define CELER_ENSURE(COND) CELER_DEBUG_ASSERT_(COND, postcondition)
#    define CELER_ASSERT_UNREACHABLE() CELER_DEBUG_FAIL_("", unreachable)
#else
#    define CELER_EXPECT(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSERT(COND) CELER_NOASSERT_(COND)
#    define CELER_ENSURE(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSERT_UNREACHABLE() CELER_UNREACHABLE
#endif

#ifndef __CUDA_ARCH__
#    define CELER_VALIDATE(COND, MSG) CELER_RUNTIME_ASSERT_(COND, MSG)
#    define CELER_NOT_CONFIGURED(WHAT) CELER_DEBUG_FAIL_(WHAT, unconfigured)
#    define CELER_NOT_IMPLEMENTED(WHAT) CELER_DEBUG_FAIL_(WHAT, unimplemented)
#else
#    define CELER_VALIDATE(COND, MSG)                                          \
        ::celeritas::throw_debug_error(::celeritas::DebugErrorType::assertion, \
                                       "CELER_VALIDATE cannot be called "      \
                                       "from device code",                     \
                                       __FILE__,                               \
                                       __LINE__)
#    define CELER_NOT_CONFIGURED(WHAT) CELER_ASSERT(0)
#    define CELER_NOT_IMPLEMENTED(WHAT) CELER_ASSERT(0)
#endif

/*!
 * \def CELER_CUDA_CALL
 *
 * When CUDA support is enabled, execute the wrapped statement and throw a
 * RuntimeError if it fails. If CUDA is disabled, throw an unconfigured
 * assertion.
 *
 * If it fails, we call \c cudaGetLastError to clear the error code.
 *
 * \code
 *     CELER_CUDA_CALL(cudaMalloc(&ptr_gpu, 100 * sizeof(float)));
 *     CELER_CUDA_CALL(cudaDeviceSynchronize());
 * \endcode
 *
 * \note A file that uses this macro must include \c cuda_runtime_api.h or be
 * compiled by NVCC (which implicitly includes that header).
 */
#if CELERITAS_USE_CUDA
#    define CELER_CUDA_CALL(STATEMENT)                       \
        do                                                   \
        {                                                    \
            cudaError_t cuda_result_ = (STATEMENT);          \
            if (CELER_UNLIKELY(cuda_result_ != cudaSuccess)) \
            {                                                \
                cudaGetLastError();                          \
                ::celeritas::throw_cuda_call_error(          \
                    cudaGetErrorString(cuda_result_),        \
                    #STATEMENT,                              \
                    __FILE__,                                \
                    __LINE__);                               \
            }                                                \
        } while (0)
#else
#    define CELER_CUDA_CALL(STATEMENT)    \
        do                                \
        {                                 \
            CELER_NOT_CONFIGURED("CUDA"); \
        } while (0)
#endif

/*!
 * \def CELER_CUDA_CHECK_ERROR
 *
 * After a kernel launch or other call, check that no CUDA errors have
 * occurred. This is also useful for checking success after external CUDA
 * libraries have been called.
 */
#define CELER_CUDA_CHECK_ERROR() CELER_CUDA_CALL(cudaPeekAtLastError())

/*!
 * \def CELER_MPI_CALL
 *
 * When MPI support is enabled, execute the wrapped statement and throw a
 * RuntimeError if it fails. If MPI is disabled, throw an unconfigured
 * assertion.
 *
 * \note A file that uses this macro must include \c mpi.h.
 */
#if CELERITAS_USE_MPI
#    define CELER_MPI_CALL(STATEMENT)                             \
        do                                                        \
        {                                                         \
            int mpi_result_ = (STATEMENT);                        \
            if (CELER_UNLIKELY(mpi_result_ != MPI_SUCCESS))       \
            {                                                     \
                ::celeritas::throw_mpi_call_error(                \
                    mpi_result_, #STATEMENT, __FILE__, __LINE__); \
            }                                                     \
        } while (0)
#else
#    define CELER_MPI_CALL(STATEMENT)    \
        do                               \
        {                                \
            CELER_NOT_CONFIGURED("MPI"); \
        } while (0)
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
// FUNCTIONS
//---------------------------------------------------------------------------//
enum class DebugErrorType
{
    precondition,  //!< Precondition contract violation
    internal,      //!< Internal assertion check failure
    unreachable,   //!< Internal assertion: unreachable code path
    unconfigured,  //!< Internal assertion: required feature not enabled
    unimplemented, //!< Internal assertion: not yet implemented
    postcondition, //!< Postcondition contract violation
};

//---------------------------------------------------------------------------//
// FUNCTIONS
//---------------------------------------------------------------------------//
// Construct and throw a DebugError.
[[noreturn]] void throw_debug_error(DebugErrorType which,
                                    const char*    condition,
                                    const char*    file,
                                    int            line);

// Construct and throw a RuntimeError for failed CUDA calls.
[[noreturn]] void throw_cuda_call_error(const char* error_string,
                                        const char* code,
                                        const char* file,
                                        int         line);

// Construct and throw a RuntimeError for failed MPI calls.
[[noreturn]] void throw_mpi_call_error(int         errorcode,
                                       const char* code,
                                       const char* file,
                                       int         line);

#ifndef __CUDA_ARCH__
// Construct and throw a RuntimeError.
[[noreturn]] void throw_runtime_error(std::string msg,
                                      const char* condition,
                                      const char* file,
                                      int         line);
#endif

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
/*!
 * Error thrown by Celeritas assertions.
 */
class DebugError : public std::logic_error
{
  public:
    // Delegating constructors
    explicit DebugError(const char* msg);
    explicit DebugError(const std::string& msg);
};

//---------------------------------------------------------------------------//
/*!
 * Error thrown by working code from unexpected runtime conditions.
 */
class RuntimeError : public std::runtime_error
{
  public:
    // Delegating constructor
    explicit RuntimeError(const char* msg);
    explicit RuntimeError(const std::string& msg);
};

#endif //__CUDA_ARCH__

//---------------------------------------------------------------------------//
} // namespace celeritas
