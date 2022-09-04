//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file Assert.hh
 * \brief Macros, exceptions, and helpers for assertions and error handling.
 *
 * This defines host- and device-compatible assertion macros that are toggled
 * on the \c CELERITAS_DEBUG configure macro.
 */
//---------------------------------------------------------------------------//
#pragma once

#include <stdexcept>
#include <string>
#if defined(__HIP_DEVICE_COMPILE__)
#    include <assert.h>
#    include <hip/hip_runtime.h>
#elif defined(__CUDA_ARCH__)
// No assert header needed for CUDA
#else
#    include <sstream>
#endif

#include "celeritas_config.h"

#include "Macros.hh"

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
/*!
 * \def CELER_EXPECT
 *
 * Precondition debug assertion macro. We "expect" that the input values
 * or initial state satisfy a precondition, and we throw exception in
 * debug mode if they do not.
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
 * message that is streamed as the second argument. This macro cannot be used
 * in \c __device__ -annotated code.
 *
 * The error message should read: \verbatim
   <PROBLEM> (<WHY IT'S A PROBLEM>) <SUGGESTION>?
  \endverbatim
 *
 * Examples with correct casing and punctuation:
 * - failed to open '{filename}' (should contain relaxation data)
 * - unexpected end of file '{filename}' (data is inconsistent with
 *   boundaries)
 * - MPI was not initialized (needed to construct a communicator). Maybe set
 *   the environment variable CELER_DISABLE_PARALLEL=1 to disable
 *   externally?"
 * - min_range={opts.min_range} (must be positive)"
 *
 * \code
 * CELER_VALIDATE(file_stream,
 *                << "failed to open '" << filename
 *                << "' (should contain relaxation data)");
 * \endcode
 *
 * An always-on debug-type assertion without a detailed message can be
 * constructed by omitting the stream (but leaving the comma):
 * \code
    CELER_VALIDATE(file_stream,);
 * \endcode
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
/*!
 * \def CELER_NOT_IMPLEMENTED
 *
 * Assert if the code point is reached because a feature has yet to be fully
 * implemented. This placeholder is so that code paths can be "declared but not
 * defined" and implementations safely postponed in a greppable manner.
 */

//! \cond
#if CELERITAS_USE_HIP || defined(NDEBUG)
// HIP "assert" can cause unexpected device failures on AMD (simultaneous
// writes from multiple threads), plus it will be disabled if NDEBUG -- same
// with CUDA
#    define CELER_DEVICE_ASSERT_(COND)                                      \
        do                                                                  \
        {                                                                   \
            if (CELER_UNLIKELY(!(COND)))                                    \
            {                                                               \
                ::celeritas::device_debug_error(#COND, __FILE__, __LINE__); \
            }                                                               \
        } while (0)
#    define CELER_DEVICE_ASSERT_UNREACHABLE_()                             \
        do                                                                 \
        {                                                                  \
            ::celeritas::device_debug_error(                               \
                "Unreachable code point encountered", __FILE__, __LINE__); \
            ::celeritas::unreachable();                                    \
        } while (0)
#elif CELERITAS_USE_CUDA && !defined(NDEBUG)
// CUDA assert macro is enabled
#    define CELER_DEVICE_ASSERT_(COND) \
        do                             \
        {                              \
            assert(COND);              \
        } while (0)
#    define CELER_DEVICE_ASSERT_UNREACHABLE_() \
        do                                     \
        {                                      \
            assert(false);                     \
            ::celeritas::unreachable();        \
        } while (0)
#endif

#define CELER_DEBUG_ASSERT_(COND, WHICH)                                       \
    do                                                                         \
    {                                                                          \
        if (CELER_UNLIKELY(!(COND)))                                           \
            throw ::celeritas::DebugError({::celeritas::DebugErrorType::WHICH, \
                                           #COND,                              \
                                           __FILE__,                           \
                                           __LINE__});                         \
    } while (0)
#define CELER_DEBUG_FAIL_(MSG, WHICH)                                       \
    do                                                                      \
    {                                                                       \
        throw ::celeritas::DebugError(                                      \
            {::celeritas::DebugErrorType::WHICH, MSG, __FILE__, __LINE__}); \
    } while (0)
#define CELER_RUNTIME_ASSERT_(COND, MSG)                              \
    do                                                                \
    {                                                                 \
        if (CELER_UNLIKELY(!(COND)))                                  \
        {                                                             \
            std::ostringstream celer_runtime_msg_;                    \
            celer_runtime_msg_ MSG;                                   \
            throw ::celeritas::RuntimeError::from_validate(           \
                celer_runtime_msg_.str(), #COND, __FILE__, __LINE__); \
        }                                                             \
    } while (0)
#define CELER_NOASSERT_(COND)   \
    do                          \
    {                           \
        if (false && (COND)) {} \
    } while (0)
//! \endcond

#if CELERITAS_DEBUG && CELER_DEVICE_COMPILE
#    define CELER_EXPECT(COND) CELER_DEVICE_ASSERT_(COND)
#    define CELER_ASSERT(COND) CELER_DEVICE_ASSERT_(COND)
#    define CELER_ENSURE(COND) CELER_DEVICE_ASSERT_(COND)
#    define CELER_ASSERT_UNREACHABLE() CELER_DEVICE_ASSERT_UNREACHABLE_()
#elif CELERITAS_DEBUG && !CELER_DEVICE_COMPILE
#    define CELER_EXPECT(COND) CELER_DEBUG_ASSERT_(COND, precondition)
#    define CELER_ASSERT(COND) CELER_DEBUG_ASSERT_(COND, internal)
#    define CELER_ENSURE(COND) CELER_DEBUG_ASSERT_(COND, postcondition)
#    define CELER_ASSERT_UNREACHABLE() CELER_DEBUG_FAIL_("", unreachable)
#else
#    define CELER_EXPECT(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSERT(COND) CELER_NOASSERT_(COND)
#    define CELER_ENSURE(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSERT_UNREACHABLE() ::celeritas::unreachable()
#endif

#if !CELER_DEVICE_COMPILE
#    define CELER_VALIDATE(COND, MSG) CELER_RUNTIME_ASSERT_(COND, MSG)
#    define CELER_NOT_CONFIGURED(WHAT) CELER_DEBUG_FAIL_(WHAT, unconfigured)
#    define CELER_NOT_IMPLEMENTED(WHAT) CELER_DEBUG_FAIL_(WHAT, unimplemented)
#else
#    define CELER_VALIDATE(COND, MSG)                                         \
        throw ::celeritas::DebugError(::celeritas::DebugErrorType::internal,  \
                                      "CELER_VALIDATE cannot be called from " \
                                      "device code",                          \
                                      __FILE__,                               \
                                      __LINE__);
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
       CELER_CUDA_CALL(cudaMalloc(&ptr_gpu, 100 * sizeof(float)));
       CELER_CUDA_CALL(cudaDeviceSynchronize());
 * \endcode
 *
 * \note A file that uses this macro must include \c
 * corecel/device_runtime_api.h or be compiled by NVCC (which implicitly
 * includes that header).
 */
#if CELERITAS_USE_CUDA
#    define CELER_CUDA_CALL(STATEMENT)                             \
        do                                                         \
        {                                                          \
            cudaError_t cuda_result_ = (STATEMENT);                \
            if (CELER_UNLIKELY(cuda_result_ != cudaSuccess))       \
            {                                                      \
                cudaGetLastError();                                \
                throw ::celeritas::RuntimeError::from_device_call( \
                    cudaGetErrorString(cuda_result_),              \
                    #STATEMENT,                                    \
                    __FILE__,                                      \
                    __LINE__);                                     \
            }                                                      \
        } while (0)
#else
#    define CELER_CUDA_CALL(STATEMENT)    \
        do                                \
        {                                 \
            CELER_NOT_CONFIGURED("CUDA"); \
        } while (0)
#endif

/*!
 * \def CELER_HIP_CALL
 *
 * When HIP support is enabled, execute the wrapped statement and throw a
 * RuntimeError if it fails. If HIP is disabled, throw an unconfigured
 * assertion.
 *
 * If it fails, we call \c hipGetLastError to clear the error code.
 *
 * \code
       CELER_HIP_CALL(hipMalloc(&ptr_gpu, 100 * sizeof(float)));
       CELER_HIP_CALL(hipDeviceSynchronize());
 * \endcode
 *
 * \note A file that uses this macro must include \c hip_runtime_api.h or be
 * compiled by NVCC (which implicitly includes that header).
 */
#if CELERITAS_USE_HIP
#    define CELER_HIP_CALL(STATEMENT)                              \
        do                                                         \
        {                                                          \
            hipError_t hip_result_ = (STATEMENT);                  \
            if (CELER_UNLIKELY(hip_result_ != hipSuccess))         \
            {                                                      \
                hipGetLastError();                                 \
                throw ::celeritas::RuntimeError::from_device_call( \
                    hipGetErrorString(hip_result_),                \
                    #STATEMENT,                                    \
                    __FILE__,                                      \
                    __LINE__);                                     \
            }                                                      \
        } while (0)
#else
#    define CELER_HIP_CALL(STATEMENT)    \
        do                               \
        {                                \
            CELER_NOT_CONFIGURED("HIP"); \
        } while (0)
#endif

/*!
 * \def CELER_DEVICE_CALL_PREFIX
 *
 * Prepend the argument with "cuda" or "hip" and call with the appropriate
 * checking statement as above.
 *
 * Example:
 *
 * \code
       CELER_DEVICE_CALL_PREFIX(Malloc(&ptr_gpu, 100 * sizeof(float)));
       CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
 * \endcode
 *
 */
#if CELERITAS_USE_CUDA
#    define CELER_DEVICE_CALL_PREFIX(STMT) CELER_CUDA_CALL(cuda##STMT)
#elif CELERITAS_USE_HIP
#    define CELER_DEVICE_CALL_PREFIX(STMT) CELER_HIP_CALL(hip##STMT)
#else
#    define CELER_DEVICE_CALL_PREFIX(STMT)       \
        do                                       \
        {                                        \
            CELER_NOT_CONFIGURED("CUDA or HIP"); \
        } while (0)
#endif

/*!
 * \def CELER_DEVICE_CHECK_ERROR
 *
 * After a kernel launch or other call, check that no CUDA errors have
 * occurred. This is also useful for checking success after external CUDA
 * libraries have been called.
 */
#define CELER_DEVICE_CHECK_ERROR() CELER_DEVICE_CALL_PREFIX(PeekAtLastError())

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
                throw ::celeritas::RuntimeError::from_mpi_call(   \
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

//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
// ENUMERATIONS
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

enum class RuntimeErrorType
{
    validate, //!< Celeritas runtime error
    device,   //!< CUDA or HIP
    mpi
};

//! Detailed properties of a debug assertion failure
struct DebugErrorDetails
{
    DebugErrorType which;
    const char*    condition;
    const char*    file;
    int            line;
};

//! Detailed properties of a runtime error
struct RuntimeErrorDetails
{
    RuntimeErrorType which{RuntimeErrorType::validate};
    std::string      what{};
    const char*      condition{nullptr};
    const char*      file{nullptr};
    int              line{0};
};

//---------------------------------------------------------------------------//
// FUNCTIONS
//---------------------------------------------------------------------------//

//! Invoke undefined behavior
[[noreturn]] inline CELER_FUNCTION void unreachable()
{
    CELER_UNREACHABLE;
}

//! Get a pretty string version of a debug error
const char* to_cstring(DebugErrorType which);

//! Get a pretty string version of a runtime error
const char* to_cstring(RuntimeErrorType which);

//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
/*!
 * Error thrown by Celeritas assertions.
 */
class DebugError : public std::logic_error
{
  public:
    // Construct from debug attributes
    explicit DebugError(DebugErrorDetails);

    //! Access the debug data
    const DebugErrorDetails& details() const { return details_; }

  private:
    DebugErrorDetails details_;
};

//---------------------------------------------------------------------------//
/*!
 * Error thrown by working code from unexpected runtime conditions.
 */
class RuntimeError : public std::runtime_error
{
  public:
    // Construct from validation failure
    static RuntimeError
    from_validate(std::string msg, const char* code, const char* file, int line);

    // Construct from device call
    static RuntimeError from_device_call(const char* error_string,
                                         const char* code,
                                         const char* file,
                                         int         line);

    // Construct from device call
    static RuntimeError
    from_mpi_call(int errorcode, const char* code, const char* file, int line);

    // Construct from details
    explicit RuntimeError(RuntimeErrorDetails);

    //! Access detailed information
    const RuntimeErrorDetails& details() const { return details_; }

  private:
    RuntimeErrorDetails details_;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//

#if CELER_DEVICE_COMPILE
__attribute__((noinline)) __host__ __device__ inline void
device_debug_error(const char* condition, const char* file, unsigned int line)
{
    printf("%s:%u:\nceleritas: internal assertion failed: %s\n",
           file,
           line,
           condition);
#    if CELERITAS_USE_CUDA
    __trap();
#    else
    abort();
#    endif
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
