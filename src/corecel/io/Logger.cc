//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <algorithm>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>  // IWYU pragma: keep
#include <string>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"

#include "ColorUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
//! Default global logger prints the error message with basic colors
void default_global_handler(Provenance prov, LogLevel lev, std::string msg)
{
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> scoped_lock{log_mutex};

    if (lev == LogLevel::debug || lev >= LogLevel::warning)
    {
        // Output problem line/file for debugging or high level
        std::clog << color_code('x') << prov.file;
        if (prov.line)
            std::clog << ':' << prov.line;
        std::clog << color_code(' ') << ": ";
    }

    // clang-format on
    std::clog << to_color_code(lev) << to_cstring(lev) << ": "
              << color_code(' ') << msg << std::endl;
}

//---------------------------------------------------------------------------//
//! Log the local node number as well as the message
class LocalHandler
{
  public:
    explicit LocalHandler(MpiCommunicator const& comm) : rank_(comm.rank()) {}

    void operator()(Provenance prov, LogLevel lev, std::string msg)
    {
        // Use buffered 'clog'
        std::clog << color_code('x') << prov.file << ':' << prov.line
                  << color_code(' ') << ": " << color_code('W') << "rank "
                  << rank_ << ": " << color_code('x') << to_cstring(lev)
                  << ": " << color_code(' ') << msg << std::endl;
    }

  private:
    int rank_;
};

//---------------------------------------------------------------------------//
/*!
 * Set the log level from an environment variable, warn on failure.
 */
void set_log_level_from_env(Logger* log, std::string const& level_env)
{
    CELER_EXPECT(log);
    try
    {
        log->level(log_level_from_env(level_env));
    }
    catch (RuntimeError const& e)
    {
        (*log)(CELER_CODE_PROVENANCE, LogLevel::warning) << e.details().what;
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with default communicator and handler.
 */
Logger::Logger(LogHandler handle)
    : Logger(MpiCommunicator::comm_default(), std::move(handle))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with communicator (only rank zero is active) and handler.
 */
Logger::Logger(MpiCommunicator const& comm, LogHandler handle)
{
    if (comm.rank() == 0)
    {
        // Accept handler, otherwise it is a "null" function pointer.
        handle_ = std::move(handle);
    }
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the log level from an environment variable.
 */
LogLevel log_level_from_env(std::string const& level_env)
{
    // Search for the provided environment variable to set the default
    // logging level using the `to_cstring` function in LoggerTypes.
    std::string const& env_value = celeritas::getenv(level_env);
    if (env_value.empty())
    {
        return Logger::default_level();
    }

    auto levels = range(LogLevel::size_);
    auto iter = std::find_if(
        levels.begin(), levels.end(), [&env_value](LogLevel lev) {
            return env_value == to_cstring(lev);
        });
    CELER_VALIDATE(iter != levels.end(),
                   << "invalid log level '" << env_value
                   << "' in environment variable '" << level_env << "'");
    return *iter;
}

//---------------------------------------------------------------------------//
/*!
 * Create a default logger using the world communicator.
 *
 * This function can be useful when resetting a test harness.
 */
Logger make_default_world_logger()
{
    Logger log{&default_global_handler};
    set_log_level_from_env(&log, "CELER_LOG");
    return log;
}

//---------------------------------------------------------------------------//
/*!
 * Create a default logger using the local communicator.
 */
Logger make_default_self_logger()
{
    auto comm = MpiCommunicator::comm_default();
    auto handler = ScopedMpiInit::status() != ScopedMpiInit::Status::disabled
                       ? LocalHandler{comm}
                       : LogHandler{&default_global_handler};
    Logger log{comm, std::move(handler)};
    set_log_level_from_env(&log, "CELER_LOG_LOCAL");
    return log;
}

//---------------------------------------------------------------------------//
/*!
 * Parallel-enabled logger: print only on "main" process.
 *
 * Setting the "CELER_LOG" environment variable to "debug", "info", "error",
 * etc. will change the default log level.
 */
Logger& world_logger()
{
    // Use the null communicator if MPI isn't enabled, otherwise comm_world
    static Logger logger = make_default_world_logger();
    return logger;
}

//---------------------------------------------------------------------------//
/*!
 * Serial logger: print on *every* process that calls it.
 *
 * Setting the "CELER_LOG_LOCAL" environment variable to "debug", "info",
 * "error", etc. will change the default log level.
 */
Logger& self_logger()
{
    // Use the null communicator if MPI isn't enabled, otherwise comm_local.
    // If only
    static Logger logger = make_default_self_logger();
    return logger;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
