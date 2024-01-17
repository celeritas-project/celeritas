//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.cc
//---------------------------------------------------------------------------//
#include "MultiExceptionHandler.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Recursively log a possibly nested exception message.
void log_exception(std::exception const& e, Logger::Message* msg)
{
    try
    {
        std::rethrow_if_nested(e);
    }
    catch (std::exception const& next)
    {
        log_exception(next, msg);
        *msg << "\n... from: ";
    }
    catch (...)
    {
        // Ignore unknown exception
    }
    *msg << e.what();
}

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
    auto exc_vec = exceptions.release();

    for (auto eptr_iter = exc_vec.begin() + 1; eptr_iter != exc_vec.end();
         ++eptr_iter)
    {
        auto msg = CELER_LOG_LOCAL(critical);
        msg << "ignoring exception: ";
        try
        {
            std::rethrow_exception(*eptr_iter);
        }
        catch (std::exception const& e)
        {
            log_exception(e, &msg);
        }
        catch (...)
        {
            msg << "unknown type";
        }
    }

    std::rethrow_exception(exc_vec.front());
}
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Terminate if destroyed without handling exceptions.
 */
MultiExceptionHandler::~MultiExceptionHandler()
{
    if (CELER_UNLIKELY(!exceptions_.empty()))
    {
        for (auto eptr : exceptions_)
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
}

//---------------------------------------------------------------------------//
/*!
 * Thread-safe capture of the given exception.
 */
void MultiExceptionHandler::operator()(std::exception_ptr p)
{
#ifdef _OPENMP
#    pragma omp critical(MultiExceptionHandler)
#endif
    {
        exceptions_.push_back(std::move(p));
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
