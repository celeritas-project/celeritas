//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <iostream>

#include "corecel/Assert.hh"

#include "LogHandlers.hh"
#include "LoggerTypes.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
auto safe_getenv_loglevel(char const* env_var, LogLevel default_level)
    -> LogLevel
{
    try
    {
        return getenv_loglevel(env_var, default_level);
    }
    catch (RuntimeError const& e)
    {
        std::clog << "Error during logger setup: " << e.what() << std::endl;
        return default_level;
    }
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct a logger with handle.
 */
Logger::Logger(LogHandler&& handle)
    : Logger(std::move(handle), default_level())
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a logger with handle and level.
 */
Logger::Logger(LogHandler&& handle, LogLevel min_lev)
    : handle_(std::move(handle)), min_level_(min_lev)
{
    CELER_EXPECT(min_level_ != LogLevel::size_);
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * App-level logger: print only on "main" process.
 *
 * Setting the "CELER_LOG" environment variable to "debug", "info",
 * "error", etc. will change the default log level.
 *
 * \sa CELER_LOG .
 */
Logger& world_logger()
{
    static Logger logger{StreamLogHandler{std::clog},
                         safe_getenv_loglevel("CELER_LOG", LogLevel::status)};
    return logger;
}

//---------------------------------------------------------------------------//
/*!
 * Serial logger: print on \em every process that calls it.
 *
 * Setting the "CELER_LOG_LOCAL" environment variable to "debug", "info",
 * "error", etc. will change the default log level.
 *
 * \sa CELER_LOG_LOCAL .
 */
Logger& self_logger()
{
    static Logger logger{
        StreamLogHandler{std::clog},
        safe_getenv_loglevel("CELER_LOG_LOCAL", LogLevel::warning)};
    return logger;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
