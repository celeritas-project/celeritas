//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/LoggerMessage.cc
//---------------------------------------------------------------------------//
#include "LoggerMessage.hh"

#include <exception>
#include <functional>
#include <sstream>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to function object, etc.
 *
 * The handle *may be* null, indicating that the output of this message will
 * not be displayed.
 */
LoggerMessage::LoggerMessage(LogHandler* handle, Provenance prov, LogLevel lev)
    : handle_(handle), prov_(prov), lev_(lev)
{
    CELER_EXPECT(!handle_ || *handle_);
    if (handle_)
    {
        // std::function is defined, so create the output stream
        os_ = std::make_unique<std::ostringstream>();
    }
    CELER_ENSURE(bool(handle_) == bool(os_));
}

//---------------------------------------------------------------------------//
/*!
 * Flush message on destruction.
 */
LoggerMessage::~LoggerMessage()
{
    if (os_)
    {
        try
        {
            auto& os = dynamic_cast<std::ostringstream&>(*os_);

            // Write to the handler
            (*handle_)(prov_, lev_, os.str());
        }
        catch (std::exception const& e)
        {
            std::cerr
                << "An error occurred writing a log message: " << e.what()
                << std::endl;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
