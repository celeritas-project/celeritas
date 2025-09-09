//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/LoggerMessage.cc
//---------------------------------------------------------------------------//
#include "LoggerMessage.hh"

#include <exception>
#include <functional>
#include <iostream>
#include <sstream>

#include "corecel/Assert.hh"

#include "../Logger.hh"
#include "../LoggerTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create the message when handle is non-null.
 *
 * This saves the provided data and allocates a stream for output.
 */
void LoggerMessage::construct_impl(LogProvenance&& prov, LogLevel lev)
{
    CELER_EXPECT(handle_ && *handle_);

    prov_ = std::move(prov);
    lev_ = lev;
    os_ = std::make_unique<std::ostringstream>();
}

//---------------------------------------------------------------------------//
/*!
 * Flush message on destruction.
 *
 * This is only called when \c os_ is nonnull.
 */
void LoggerMessage::destroy_impl() noexcept
{
    try
    {
        // Write to the handler
        (*handle_)(prov_, lev_, std::move(*os_).str());
    }
    catch (std::exception const& e)
    {
        std::cerr << "An error occurred writing a log message: " << e.what()
                  << std::endl;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
