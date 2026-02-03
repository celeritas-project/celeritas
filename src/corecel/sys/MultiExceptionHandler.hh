//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>
#include <utility>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Temporarily store exception pointers to allow propagation across threads.
 *
 * This is useful for storing multiple exceptions in unrelated loops (where one
 * exception shouldn't affect the program flow outside of the scope),
 * especially for OpenMP parallel execution, where exceptions cannot be
 * propagated.
 * \code
    MultiExceptionHandler capture_exception;
    #pragma omp parallel for
    for (size_type i = 0; i < data.states.size(); ++i)
    {
        CELER_TRY_HANDLE(step(TrackSlotId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
 * \endcode
 *
 * If an exception is caught and the stored exception pointers are not cleared
 * via \c release or \c log_and_rethrow , then at the end of the exception
 * handler's lifetime, the stored exceptions will be printed and the program
 * will be terminated.
 *
 * \note This class implements an OpenMP \c critical mutex, not a \c std
 * mutex. If using this class in a \c std::thread context, wrap the call
 * operator in a lambda with a \c std::scoped_lock . We could refactor as a
 * CRTP class with a protected \c push_back function that lets us specialize
 * the mutex implementation.
 */
class MultiExceptionHandler
{
  public:
    //!@{
    //! \name Type aliases
    using VecExceptionPtr = std::vector<std::exception_ptr>;
    //!@}

  public:
    // Default all construct/copy/move
    MultiExceptionHandler() = default;
    CELER_DEFAULT_COPY_MOVE(MultiExceptionHandler);

    // Terminate if destroyed without handling exceptions
    inline ~MultiExceptionHandler() noexcept(!CELERITAS_DEBUG);

    // Thread-safe capture of the given exception
    void operator()(std::exception_ptr p);

    //! Whether no exceptions have been stored (not thread safe)
    bool empty() const { return exceptions_.empty(); }

    //! Release exceptions for someone else to process (not thread safe)
    VecExceptionPtr release() && { return std::move(exceptions_); }

  private:
    VecExceptionPtr exceptions_;

    [[noreturn]] void log_and_terminate() const;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
namespace detail
{
// Private implementation function for throwing exceptions
[[noreturn]] void log_and_rethrow_impl(MultiExceptionHandler&& exceptions);
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Throw the first exception and log all the rest.
 *
 * All logged exceptions will be thrown using \c CELER_LOG_LOCAL(critical) .
 */
inline void log_and_rethrow(MultiExceptionHandler&& exceptions)
{
    if (CELER_UNLIKELY(!exceptions.empty()))
    {
        detail::log_and_rethrow_impl(std::move(exceptions));
    }
}

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Terminate if destroyed without handling exceptions.
 */
MultiExceptionHandler::~MultiExceptionHandler() noexcept(!CELERITAS_DEBUG)
{
    if (CELER_UNLIKELY(!exceptions_.empty()))
    {
        this->log_and_terminate();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
