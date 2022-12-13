//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>
#include <vector>

#include "corecel/Macros.hh"

/*!
 * \def CELER_TRY_ELSE
 *
 * Execute the statement, catching *all* thrown errors by calling the given
 * function-like operator with a \c std::exception_ptr object.
 */
#define CELER_TRY_ELSE(STATEMENT, HANDLE_EXCEPTION) \
    try                                             \
    {                                               \
        STATEMENT;                                  \
    }                                               \
    catch (...)                                     \
    {                                               \
        HANDLE_EXCEPTION(std::current_exception()); \
    }

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Temporarily store exception pointers.
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
        CELER_TRY_ELSE(launch(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
 * \endcode
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
    MultiExceptionHandler()                                        = default;
    MultiExceptionHandler(MultiExceptionHandler&&)                 = default;
    MultiExceptionHandler& operator=(MultiExceptionHandler&&)      = default;
    MultiExceptionHandler(const MultiExceptionHandler&)            = default;
    MultiExceptionHandler& operator=(const MultiExceptionHandler&) = default;

    // Terminate if destroyed without handling exceptions
    ~MultiExceptionHandler();

    // Thread-safe capture of the given exception
    void operator()(std::exception_ptr p);

    //! Whether no exceptions have been stored (not thread safe)
    bool empty() const { return exceptions_.empty(); }

    //! Release exceptions for someone else to process (not thread safe)
    VecExceptionPtr release() { return std::move(exceptions_); }

  private:
    VecExceptionPtr exceptions_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
namespace detail
{
// Private implementation function for throwing exceptions
[[noreturn]] void log_and_rethrow_impl(MultiExceptionHandler&& exceptions);
} // namespace detail

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
} // namespace celeritas
