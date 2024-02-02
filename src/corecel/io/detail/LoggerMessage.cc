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
 * Create the message when handle is non-null.
 */
void LoggerMessage::construct_impl(Provenance&& prov, LogLevel lev)
{
    CELER_EXPECT(handle_ && *handle_);
    lev_ = lev;

    // std::function is defined, so create the output stream
    os_ = std::make_unique<std::ostringstream>();
    prov_ = std::move(prov);
}

//---------------------------------------------------------------------------//
/*!
 * Flush message on destruction.
 *
 * This is only called when \c os_ is nonzero.
 */
void LoggerMessage::destroy_impl() noexcept
{
    try
    {
        auto& os = dynamic_cast<std::ostringstream&>(*os_);

        // Write to the handler
        (*handle_)(prov_, lev_, os.str());
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
