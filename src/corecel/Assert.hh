//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file Assert.hh
 * \brief Macros, exceptions, and helpers for assertions and error handling.
 *
 * This defines host- and device-compatible assertion macros that are toggled
 * on the \c CELERITAS_DEBUG and \c CELERITAS_DEVICE_DEBUG configure macros.
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

#include "corecel/Config.hh"

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
 * handling code (e.g. in \c std::visit by assuming
 * \code !var_obj.valueless_by_exception() \endcode ).
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
 * - \"failed to open '{filename}' (should contain relaxation data)\"
 * - \"unexpected end of file '{filename}' (data is inconsistent with
 *   boundaries)\"
 * - \"MPI was not initialized (needed to construct a communicator). Maybe set
 *   the environment variable CELER_DISABLE_PARALLEL=1 to disable externally?\"
 * - \"invalid min_range={opts.min_range} (must be positive)\"
 *
 * This looks in practice like:
 * \code
   CELER_VALIDATE(file_stream,
                  << "failed to open '" << filename
                  << "' (should contain relaxation data)");
 * \endcode
 */
/*!
 * \def CELER_DEBUG_FAIL
 *
 * Throw a debug assertion regardless of the \c CELERITAS_DEBUG setting. This
 * is used internally but is also useful for catching subtle programming errors
 * in downstream code.
 */
/*!
 * \def CELER_ASSERT_UNREACHABLE
 *
 * Throw an assertion if the code point is reached. When debug assertions are
 * turned off, this changes to a compiler hint that improves optimization (and
 * may force the code to exit uncermoniously if the point is encountered,
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
        ::celeritas::throw_debug_error(    \
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

#define CELER_DEBUG_FAIL(MSG, WHICH)    \
    do                                  \
    {                                   \
        CELER_DEBUG_THROW_(MSG, WHICH); \
        ::celeritas::unreachable();     \
    } while (0)

#if !CELER_DEVICE_COMPILE
#    define CELER_RUNTIME_THROW(WHICH, WHAT, COND) \
        ::celeritas::throw_runtime_error({         \
            WHICH,                                 \
            WHAT,                                  \
            COND,                                  \
            __FILE__,                              \
            __LINE__,                              \
        })
#elif CELERITAS_DEBUG
#    define CELER_RUNTIME_THROW(WHICH, WHAT, COND)                           \
        CELER_DEBUG_FAIL("Runtime errors cannot be thrown from device code", \
                         unreachable);
#else
// Avoid printf statements which can add substantially to local memory
#    define CELER_RUNTIME_THROW(WHICH, WHAT, COND) ::celeritas::unreachable()
#endif

#if (CELERITAS_DEBUG && !CELER_DEVICE_COMPILE) \
    || (CELERITAS_DEVICE_DEBUG && CELER_DEVICE_COMPILE)
#    define CELER_EXPECT(COND) CELER_DEBUG_ASSERT_(COND, precondition)
#    define CELER_ASSERT(COND) CELER_DEBUG_ASSERT_(COND, internal)
#    define CELER_ENSURE(COND) CELER_DEBUG_ASSERT_(COND, postcondition)
#    define CELER_ASSUME(COND) CELER_DEBUG_ASSERT_(COND, assumption)
#    define CELER_ASSERT_UNREACHABLE() \
        CELER_DEBUG_FAIL("unreachable code point encountered", unreachable)
#else
#    define CELER_EXPECT(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSERT(COND) CELER_NOASSERT_(COND)
#    define CELER_ENSURE(COND) CELER_NOASSERT_(COND)
#    define CELER_ASSUME(COND) CELER_NDEBUG_ASSUME_(COND)
#    define CELER_ASSERT_UNREACHABLE() ::celeritas::unreachable()
#endif

#if !CELER_DEVICE_COMPILE
#    define CELER_VALIDATE(COND, MSG)                            \
        do                                                       \
        {                                                        \
            if (CELER_UNLIKELY(!(COND)))                         \
            {                                                    \
                std::ostringstream celer_runtime_msg_;           \
                celer_runtime_msg_ MSG;                          \
                CELER_RUNTIME_THROW(                             \
                    ::celeritas::RuntimeError::validate_err_str, \
                    celer_runtime_msg_.str(),                    \
                    #COND);                                      \
            }                                                    \
        } while (0)
#else
#    define CELER_VALIDATE(COND, MSG)                \
        do                                           \
        {                                            \
            CELER_DISCARD(COND);                     \
            CELER_RUNTIME_THROW(nullptr, "", #COND); \
        } while (0)
#endif

#define CELER_NOT_CONFIGURED(WHAT) \
    CELER_RUNTIME_THROW(           \
        ::celeritas::RuntimeError::not_config_err_str, WHAT, {})
#define CELER_NOT_IMPLEMENTED(WHAT) \
    CELER_RUNTIME_THROW(::celeritas::RuntimeError::not_impl_err_str, WHAT, {})

/*!
 * \def CELER_DEVICE_API_CALL
 *
 * Safely and portably dispatch a CUDA/HIP API call.
 *
 * When CUDA or HIP support is enabled, execute the wrapped statement
 * prepend the argument with "cuda" or "hip" and throw a
 * RuntimeError if it fails. If no device platform is enabled, throw an
 * unconfigured assertion.
 *
 * \par Example:
 * \code
   CELER_DEVICE_API_CALL(Malloc(&ptr_gpu, 100 * sizeof(float)));
   CELER_DEVICE_API_CALL(DeviceSynchronize());
 * \endcode
 *
 * \note A file that uses this macro must include \c
 * corecel/DeviceRuntimeApi.hh . The \c CorecelDeviceRuntimeApiHh
 * declaration enforces this when CUDA/HIP are disabled, and the absence of
 * \c CELER_DEVICE_API_SYMBOL enforces when enabled.
 */
#if CELERITAS_USE_CUDA || CELERITAS_USE_HIP
#    define CELER_DEVICE_API_CALL(STMT)                                      \
        do                                                                   \
        {                                                                    \
            using ErrT_ = CELER_DEVICE_API_SYMBOL(Error_t);                  \
            ErrT_ result_ = CELER_DEVICE_API_SYMBOL(STMT);                   \
            if (CELER_UNLIKELY(result_ != CELER_DEVICE_API_SYMBOL(Success))) \
            {                                                                \
                result_ = CELER_DEVICE_API_SYMBOL(GetLastError)();           \
                CELER_RUNTIME_THROW(                                         \
                    CELER_DEVICE_PLATFORM_UPPER_STR,                         \
                    CELER_DEVICE_API_SYMBOL(GetErrorString)(result_),        \
                    #STMT);                                                  \
            }                                                                \
        } while (0)
#else
#    define CELER_DEVICE_API_CALL(STMT)              \
        do                                           \
        {                                            \
            CELER_NOT_CONFIGURED("CUDA or HIP");     \
            CELER_DISCARD(CorecelDeviceRuntimeApiHh) \
        } while (0)
#endif

// DEPRECATED: remove in v1.0
#define CELER_DEVICE_PREFIX(STMT) CELER_DEVICE_API_SYMBOL(STMT)
#define CELER_DEVICE_CALL_PREFIX(STMT) CELER_DEVICE_API_CALL(STMT)
#define CELER_DEVICE_CHECK_ERROR() CELER_DEVICE_API_CALL(PeekAtLastError())

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
#    define CELER_MPI_CALL(STATEMENT)                                     \
        do                                                                \
        {                                                                 \
            int mpi_result_ = (STATEMENT);                                \
            if (CELER_UNLIKELY(mpi_result_ != MPI_SUCCESS))               \
            {                                                             \
                CELER_RUNTIME_THROW(                                      \
                    "MPI", mpi_error_to_string(mpi_result_), #STATEMENT); \
            }                                                             \
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
    postcondition,  //!< Postcondition contract violation
    assumption,  //!< "Assume" violation
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
    char const* which{nullptr};  //!< Type of error (runtime, Geant4, MPI)
    std::string what{};  //!< Descriptive message
    std::string condition{};  //!< Code/test that failed
    std::string file{};  //!< Source file
    int line{0};  //!< Source line
};

//---------------------------------------------------------------------------//
// FUNCTIONS
//---------------------------------------------------------------------------//

//! Invoke undefined behavior
[[noreturn]] inline CELER_FUNCTION void unreachable()
{
    CELER_UNREACHABLE;
}

// Get a pretty string version of a debug error
char const* to_cstring(DebugErrorType which);

// Get an MPI error string
std::string mpi_error_to_string(int);

// Throw a debug error
[[noreturn]] void throw_debug_error(DebugErrorDetails&&);

// Throw a runtime error
[[noreturn]] void throw_runtime_error(RuntimeErrorDetails&&);

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
    explicit DebugError(DebugErrorDetails&&);
    CELER_DEFAULT_COPY_MOVE(DebugError);

    // Default destructor to anchor vtable
    ~DebugError() override;

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
    // Construct from details
    explicit RuntimeError(RuntimeErrorDetails&&);
    CELER_DEFAULT_COPY_MOVE(RuntimeError);

    // Default destructor to anchor vtable
    ~RuntimeError() override;

    //! Access detailed information
    RuntimeErrorDetails const& details() const { return details_; }

    //!@{
    //! String constants for "which" error message
    static char const validate_err_str[];
    static char const not_config_err_str[];
    static char const not_impl_err_str[];
    //!@}

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
    return ::celeritas::throw_debug_error({which, condition, file, line});
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
