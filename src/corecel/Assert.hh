//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#if defined(__HIP__)
#    include <hip/hip_runtime.h>
#elif defined(__CUDA_ARCH__)
// No assert header needed for CUDA
#else
#    include <ostream>  // IWYU pragma: export
#    include <sstream>  // IWYU pragma: keep
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
 * \def CELER_ASSUME
 *
 * Always-on compiler assumption. This should be used very rarely: you should
 * make sure the resulting assembly is simplified in optimize mode from using
 * the assumption. For example, sometimes informing the compiler of an
 * assumption can reduce code bloat by skipping standard library exception
 * handling code (e.g. in \c std::visit by assuming \c
 * !var_obj.valueless_by_exception() ).
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
   "<PROBLEM> (<WHY IT'S A PROBLEM>) <SUGGESTION>?"
  \endverbatim
 *
 * Examples with correct casing and punctuation:
 * - "failed to open '{filename}' (should contain relaxation data)"
 * - "unexpected end of file '{filename}' (data is inconsistent with
 *   boundaries)"
 * - "MPI was not initialized (needed to construct a communicator). Maybe set
 *   the environment variable CELER_DISABLE_PARALLEL=1 to disable
 *   externally?"
 * - "invalid min_range={opts.min_range} (must be positive)"
 *
 * This looks in pracice like:
 * \code
   CELER_VALIDATE(file_stream,
                  << "failed to open '" << filename
                  << "' (should contain relaxation data)");
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
 * \endcode
 */
/*!
 * \def CELER_NOT_IMPLEMENTED
 *
 * Assert if the code point is reached because a feature has yet to be fully
 * implemented.
 *
 * This placeholder is so that code paths can be "declared but not defined" and
 * implementations safely postponed in a greppable manner. This should \em not
 * be used to define "unused" overrides for virtual classes. A correct use case
 * would be:
 * \code
   if (z > AtomicNumber{26})
   {
       CELER_NOT_IMPLEMENTED("physics for heavy nuclides");
   }
 * \endcode
 */

//! \cond

#if !defined(__HIP__) && !defined(__CUDA_ARCH__)
// Throw in host code
#    define CELER_DEBUG_THROW_(MSG, WHICH) \
        throw ::celeritas::DebugError(     \
            {::celeritas::DebugErrorType::WHICH, MSG, __FILE__, __LINE__})
#elif defined(__CUDA_ARCH__) && !defined(NDEBUG)
// Use the assert macro for CUDA when supported
#    define CELER_DEBUG_THROW_(MSG, WHICH) \
        assert(false && sizeof(#WHICH ": " MSG))
#else
// Use a special device function to emulate assertion failure if HIP
// (assertion from multiple threads simultaeously can cause unexpected device
// failures on AMD hardware) or if NDEBUG is in use with CUDA
#    define CELER_DEBUG_THROW_(MSG, WHICH) \
        ::celeritas::device_debug_error(   \
            ::celeritas::DebugErrorType::WHICH, MSG, __FILE__, __LINE__)
#endif

#define CELER_DEBUG_ASSERT_(COND, WHICH)      \
    do                                        \
    {                                         \
        if (CELER_UNLIKELY(!(COND)))          \
        {                                     \
            CELER_DEBUG_THROW_(#COND, WHICH); \
        }                                     \
    } while (0)
#define CELER_DEBUG_FAIL_(MSG, WHICH)   \
    do                                  \
    {                                   \
        CELER_DEBUG_THROW_(MSG, WHICH); \
        ::celeritas::unreachable();     \
    } while (0)
#define CELER_NDEBUG_ASSUME_(COND)      \
    do                                  \
    {                                   \
        if (CELER_UNLIKELY(!(COND)))    \
        {                               \
            ::celeritas::unreachable(); \
        }                               \
    } while (0)
#define CELER_NOASSERT_(COND)   \
    do                          \
    {                           \
        if (false && (COND)) {} \
    } while (0)
//! \endcond

#if CELERITAS_DEBUG
#    define CELER_EXPECT(COND) CELER_DEBUG_ASSERT_(COND, precondition)
#    define CELER_ASSERT(COND) CELER_DEBUG_ASSERT_(COND, internal)
#    define CELER_ENSURE(COND) CELER_DEBUG_ASSERT_(COND, postcondition)
#    define CELER_ASSUME(COND) CELER_DEBUG_ASSERT_(COND, assumption)
#    define CELER_ASSERT_UNREACHABLE() \
        CELER_DEBUG_FAIL_("unreachable code point encountered", unreachable)
#else
#    define CELER_EXPECT(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSERT(COND) CELER_NOASSERT_(COND)
#    define CELER_ENSURE(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSUME(COND) CELER_NDEBUG_ASSUME_(COND)
#    define CELER_ASSERT_UNREACHABLE() ::celeritas::unreachable()
#endif

#if !CELER_DEVICE_COMPILE || defined(__HIP__)
#    define CELER_VALIDATE(COND, MSG)                                     \
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
#else
#    define CELER_VALIDATE(COND, MSG)                                         \
        CELER_DEBUG_FAIL_("CELER_VALIDATE cannot be called from device code", \
                          unreachable);
#endif

#define CELER_NOT_CONFIGURED(WHAT) CELER_DEBUG_FAIL_(WHAT, unconfigured)
#define CELER_NOT_IMPLEMENTED(WHAT) CELER_DEBUG_FAIL_(WHAT, unimplemented)

/*!
 * \def CELER_CUDA_CALL
 *
 * When CUDA support is enabled, execute the wrapped statement and throw a
 * RuntimeError if it fails. If CUDA is disabled, throw an unconfigured
 * assertion.
 *
 * If it fails, we call \c cudaGetLastError to clear the error code. Note that
 * this will \em not clear the code in a few fatal error cases (kernel
 * assertion failure, invalid memory access) and all subsequent CUDA calls will
 * fail.
 *
 * \code
   CELER_CUDA_CALL(cudaMalloc(&ptr_gpu, 100 * sizeof(float)));
   CELER_CUDA_CALL(cudaDeviceSynchronize());
 * \endcode
 *
 * \note A file that uses this macro must include \c
 * corecel/device_runtime_api.h .
 */
#if CELERITAS_USE_CUDA
#    define CELER_CUDA_CALL(STATEMENT)                             \
        do                                                         \
        {                                                          \
            cudaError_t cuda_result_ = (STATEMENT);                \
            if (CELER_UNLIKELY(cuda_result_ != cudaSuccess))       \
            {                                                      \
                cuda_result_ = cudaGetLastError();                 \
                throw ::celeritas::RuntimeError::from_device_call( \
                    cudaGetErrorString(cuda_result_),              \
                    #STATEMENT,                                    \
                    __FILE__,                                      \
                    __LINE__);                                     \
            }                                                      \
        } while (0)
#else
#    define CELER_CUDA_CALL(STATEMENT)                     \
        do                                                 \
        {                                                  \
            CELER_NOT_CONFIGURED("CUDA");                  \
            CELER_DISCARD(celeritas_device_runtime_api_h_) \
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
 * \note A file that uses this macro must include \c
 * corecel/device_runtime_api.h . The \c celeritas_device_runtime_api_h_
 * declaration enforces this when HIP is disabled.
 */
#if CELERITAS_USE_HIP
#    define CELER_HIP_CALL(STATEMENT)                              \
        do                                                         \
        {                                                          \
            hipError_t hip_result_ = (STATEMENT);                  \
            if (CELER_UNLIKELY(hip_result_ != hipSuccess))         \
            {                                                      \
                hip_result_ = hipGetLastError();                   \
                throw ::celeritas::RuntimeError::from_device_call( \
                    hipGetErrorString(hip_result_),                \
                    #STATEMENT,                                    \
                    __FILE__,                                      \
                    __LINE__);                                     \
            }                                                      \
        } while (0)
#else
#    define CELER_HIP_CALL(STATEMENT)                      \
        do                                                 \
        {                                                  \
            CELER_NOT_CONFIGURED("HIP");                   \
            CELER_DISCARD(celeritas_device_runtime_api_h_) \
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
 * \note A file that uses this macro must include \c
 * corecel/device_runtime_api.h . The \c celeritas_device_runtime_api_h_
 * declaration enforces this when CUDA/HIP are disabled.
 */
#if CELERITAS_USE_CUDA
#    define CELER_DEVICE_CALL_PREFIX(STMT) CELER_CUDA_CALL(cuda##STMT)
#elif CELERITAS_USE_HIP
#    define CELER_DEVICE_CALL_PREFIX(STMT) CELER_HIP_CALL(hip##STMT)
#else
#    define CELER_DEVICE_CALL_PREFIX(STMT)                 \
        do                                                 \
        {                                                  \
            CELER_NOT_CONFIGURED("CUDA or HIP");           \
            CELER_DISCARD(celeritas_device_runtime_api_h_) \
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
    internal,  //!< Internal assertion check failure
    unreachable,  //!< Internal assertion: unreachable code path
    unconfigured,  //!< Internal assertion: required feature not enabled
    unimplemented,  //!< Internal assertion: not yet implemented
    postcondition,  //!< Postcondition contract violation
    assumption,  //!< "Assume" violation
};

enum class RuntimeErrorType
{
    validate,  //!< Celeritas runtime error
    device,  //!< CUDA or HIP
    mpi,  //!< Coarse-grain parallelism
    geant,  //!< Error from Geant4
    root  //!< Error from ROOT
};

//! Detailed properties of a debug assertion failure
struct DebugErrorDetails
{
    DebugErrorType which;
    char const* condition;
    char const* file;
    int line;
};

//! Detailed properties of a runtime error
struct RuntimeErrorDetails
{
    RuntimeErrorType which{RuntimeErrorType::validate};
    std::string what{};
    char const* condition{nullptr};
    char const* file{nullptr};
    int line{0};
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
char const* to_cstring(DebugErrorType which);

//! Get a pretty string version of a runtime error
char const* to_cstring(RuntimeErrorType which);

//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
// Forward declaration of simple struct with pointer to an NLJSON object
struct JsonPimpl;

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
    DebugErrorDetails const& details() const { return details_; }

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
    from_validate(std::string msg, char const* code, char const* file, int line);

    // Construct from device call
    static RuntimeError from_device_call(char const* error_string,
                                         char const* code,
                                         char const* file,
                                         int line);

    // Construct from MPI call
    static RuntimeError
    from_mpi_call(int errorcode, char const* code, char const* file, int line);

    // Construct from call to Geant4
    static RuntimeError from_geant_exception(char const* origin,
                                             char const* code,
                                             char const* desc);

    // Construct from call to ROOT
    static RuntimeError from_root_error(char const* origin, char const* msg);

    // Construct from details
    explicit RuntimeError(RuntimeErrorDetails);

    //! Access detailed information
    RuntimeErrorDetails const& details() const { return details_; }

  private:
    RuntimeErrorDetails details_;
};

//---------------------------------------------------------------------------//
/*!
 * Base class for writing arbitrary exception context to JSON.
 *
 * This can be overridden in higher-level parts of the code for specific needs
 * (e.g., writing thread, event, and track contexts in Celeritas solver
 * kernels). Note that in order for derived classes to work with
 * `std::throw_with_nested`, they *MUST NOT* be `final`.
 */
class RichContextException : public std::exception
{
  public:
    //! Write output to the given JSON object
    virtual void output(JsonPimpl*) const = 0;

    //! Provide the name for this exception class
    virtual char const* type() const = 0;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//

#if defined(__CUDA_ARCH__) && defined(NDEBUG)
//! Host+device definition for CUDA when \c assert is unavailable
inline __attribute__((noinline)) __host__ __device__ void device_debug_error(
    DebugErrorType, char const* condition, char const* file, int line)
{
    printf("%s:%u:\nceleritas: internal assertion failed: %s\n",
           file,
           line,
           condition);
    __trap();
}
#elif defined(__HIP__)
//! Host-only HIP call (whether or not NDEBUG is in use)
inline __host__ void device_debug_error(DebugErrorType which,
                                        char const* condition,
                                        char const* file,
                                        int line)
{
    throw DebugError({which, condition, file, line});
}

//! Device-only call for HIP (must always be declared; only used if
//! NDEBUG)
inline __attribute__((noinline)) __device__ void device_debug_error(
    DebugErrorType, char const* condition, char const* file, int line)
{
    printf("%s:%u:\nceleritas: internal assertion failed: %s\n",
           file,
           line,
           condition);
    abort();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
