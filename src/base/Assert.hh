//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Assert.hh
//---------------------------------------------------------------------------//
#ifndef base_Assert_hh
#define base_Assert_hh

#include "celeritas_config.h"
#include "Macros.hh"
#ifndef __CUDA_ARCH__
#    include <stdexcept>
#endif

//---------------------------------------------------------------------------//
/*!
 * \def REQUIRE
 *
 * Precondition debug assertion macro. It is to "require" that the input values
 * or initial state satisfy a precondition.
 */
/*!
 * \def CHECK
 *
 * Internal debug assertion macro. This replaces standard \c assert usage.
 */
/*!
 * \def ENSURE
 *
 * Postcondition debug assertion macro. Use to "ensure" that return values or
 * side effects are as expected when leaving a function.
 */
#define CELER_CUDA_ASSERT_(COND) \
    do                           \
    {                            \
        assert(COND);            \
    } while (0)
#define CELER_ASSERT_(COND)                                            \
    do                                                                 \
    {                                                                  \
        if (CELER_UNLIKELY(!(COND)))                                   \
            ::celeritas::throw_debug_error(#COND, __FILE__, __LINE__); \
    } while (0)
#define CELER_NOASSERT_(COND)   \
    do                          \
    {                           \
        if (false && (COND)) {} \
    } while (0)

#if CELERITAS_DEBUG && defined(__CUDA_ARCH__)
#    define REQUIRE(x) CELER_CUDA_ASSERT_(x)
#    define CHECK(x) CELER_CUDA_ASSERT_(x)
#    define ENSURE(x) CELER_CUDA_ASSERT_(x)
#elif CELERITAS_DEBUG && !defined(__CUDA_ARCH__)
#    define REQUIRE(x) CELER_ASSERT_(x)
#    define CHECK(x) CELER_ASSERT_(x)
#    define ENSURE(x) CELER_ASSERT_(x)
#else
#    define REQUIRE(x) CELER_NOASSERT_(x)
#    define CHECK(x) CELER_NOASSERT_(x)
#    define ENSURE(x) CELER_NOASSERT_(x)
#endif

//---------------------------------------------------------------------------//
/*!
 * \def CELER_CUDA_CALL
 *
 * Execute the wrapped statement and throw a message if it fails.
 *
 * If it fails, we call \c cudaGetLastError to clear the error code.
 *
 * \code
 *     CELER_CUDA_CALL(cudaMalloc(&ptr_gpu, 100 * sizeof(float)));
 *     CELER_CUDA_CALL(cudaDeviceSynchronize());
 * \endcode
 */
#define CELER_CUDA_CALL(STATEMENT)                       \
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
    } while (0);

/*!
 * \def CELER_CUDA_CHECK_ERROR
 *
 * After a kernel launch or other call, check that no CUDA errors have
 * occurred. This is also useful for checking success after external CUDA
 * libraries have been called.
 */
#define CELER_CUDA_CHECK_ERROR() CELER_CUDA_CALL(cudaPeekAtLastError())

namespace celeritas
{
//---------------------------------------------------------------------------//
[[noreturn]] void
throw_debug_error(const char* condition, const char* file, int line);

//---------------------------------------------------------------------------//
[[noreturn]] void throw_cuda_call_error(const char* error_string,
                                        const char* code,
                                        const char* file,
                                        int         line);

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Error thrown by celeritas assertions.
 */
class DebugError : public std::logic_error
{
  public:
    explicit DebugError(const char* msg);
    explicit DebugError(const std::string& msg);
};

//---------------------------------------------------------------------------//
/*!
 * Error thrown by CELER_CUDA assertion macros.
 */
class CudaCallError : public std::runtime_error
{
  public:
    explicit CudaCallError(const char* msg);
    explicit CudaCallError(const std::string& msg);
};

#endif

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // base_Assert_hh
