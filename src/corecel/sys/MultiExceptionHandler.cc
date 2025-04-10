//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.cc
//---------------------------------------------------------------------------//
#include "MultiExceptionHandler.hh"

#include <cstring>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Helper class to unwrap nested exceptions into a vector of messages.
 *
 * Because \c rethrow_if_nested and \c rethrow_exception do not specify whether
 * the original exception (or a copy) is being thrown, it may not be safe to
 * keep a c-string reference to e.what().
 */
class ExceptionStackUnwinder
{
  public:
    using VecStr = std::vector<std::string>;

    // Default constructor
    ExceptionStackUnwinder() { messages_.reserve(2); }

    // Extract messages from an exception and all its nested exceptions
    VecStr const& operator()(std::exception const& e)
    {
        messages_.clear();
        unwind_exception(e);
        CELER_ENSURE(!messages_.empty());
        return messages_;
    }

  private:
    VecStr messages_;

    // Recursively extract messages from nested exceptions
    void unwind_exception(std::exception const& e)
    {
        try
        {
            std::rethrow_if_nested(e);
        }
        catch (std::exception const& next)
        {
            // Process deeper exceptions first
            unwind_exception(next);
        }
        // NOLINTNEXTLINE(bugprone-empty-catch)
        catch (...)
        {
            // Ignore unknown exception
            messages_.push_back("unknown exception type");
        }

        // Add current exception message after processing nested exceptions
        messages_.push_back(e.what());
    }
};

//! Helper class to manage suppression of similar error messages
class ExceptionLogger
{
  public:
    using VecStr = std::vector<std::string>;

    void operator()(VecStr const& msg_stack)
    {
        CELER_EXPECT(!msg_stack.empty());

        if (msg_stack.front() == last_msg_)
        {
            ++ignored_counter_;
        }
        else
        {
            flush_suppressed();
            last_msg_ = msg_stack.front();

            auto log_msg = CELER_LOG_LOCAL(critical);
            log_msg << "Suppressed exception from parallel thread: "
                    << join(msg_stack.begin(), msg_stack.end(), "\n... from ");
        }
    }

    // Flush any suppressed messages before destruction
    ~ExceptionLogger() noexcept
    {
        try
        {
            flush_suppressed();
        }
        // NOLINTNEXTLINE(bugprone-empty-catch)
        catch (...)
        {
        }
    }

  private:
    std::string last_msg_;
    size_type ignored_counter_{0};

    void flush_suppressed()
    {
        if (ignored_counter_ > 0)
        {
            CELER_LOG_LOCAL(warning)
                << "Suppressed " << ignored_counter_ << " similar exceptions";
            ignored_counter_ = 0;
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Throw the first exception and log all the rest.
 */
[[noreturn]] void log_and_rethrow_impl(MultiExceptionHandler&& exceptions)
{
    CELER_EXPECT(!exceptions.empty());
    auto exc_vec = std::move(exceptions).release();

    ExceptionStackUnwinder unwind_stack;
    ExceptionLogger log_exception;

    for (auto eptr_iter = exc_vec.begin() + 1; eptr_iter != exc_vec.end();
         ++eptr_iter)
    {
        try
        {
            std::rethrow_exception(*eptr_iter);
        }
        catch (std::exception const& e)
        {
            // Get error messages, deepest first
            auto const& message_stack = unwind_stack(e);
            // Log non-duplicate messages
            log_exception(message_stack);
        }
    }

    std::rethrow_exception(exc_vec.front());
}
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Terminate when destroyed without handling exceptions.
 */
[[noreturn]] void MultiExceptionHandler::log_and_terminate() const
{
    CELER_EXPECT(!exceptions_.empty());

    for (auto const& eptr : exceptions_)
    {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch (std::exception const& e)
        {
            CELER_LOG_LOCAL(critical) << e.what();
        }
        catch (...)
        {
            CELER_LOG_LOCAL(critical) << "(unknown exception)";
        }
    }
    CELER_LOG(critical) << "failed to clear exceptions from "
                           "MultiExceptionHandler";
    std::terminate();
}

//---------------------------------------------------------------------------//
/*!
 * Thread-safe capture of the given exception.
 */
void MultiExceptionHandler::operator()(std::exception_ptr p)
{
#if CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp critical(MultiExceptionHandler)
#endif
    {
        exceptions_.push_back(std::move(p));
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
