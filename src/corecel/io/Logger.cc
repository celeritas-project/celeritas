//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"

#include "ColorUtils.hh"
#include "LogHandlers.hh"
#include "LoggerTypes.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
//! Log the local node number as well as the message
class LocalHandler
{
  public:
    explicit LocalHandler(MpiCommunicator const& comm) : rank_(comm.rank()) {}

    void operator()(LogProvenance prov, LogLevel lev, std::string msg)
    {
        // Buffer to reduce I/O contention in MPI runner
        std::ostringstream os;
        os << color_code('W') << "rank " << rank_ << ": ";
        StreamLogHandler{os}(prov, lev, msg);

        std::clog << std::move(os).str();
    }

  private:
    int rank_;
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Create a logger from a handle and level environment variable.
 */
Logger
Logger::from_handle_env(LogHandler&& handle, std::string const& level_env)
{
    Logger result{std::move(handle)};

    try
    {
        result.level(log_level_from_env(level_env));
    }
    catch (RuntimeError const& e)
    {
        result(CELER_CODE_PROVENANCE, LogLevel::warning) << e.details().what;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with handler.
 *
 * A null handler will silence the logger.
 */
Logger::Logger(LogHandler&& handle) : handle_{std::move(handle)} {}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the log level from an environment variable.
 *
 * Default to \c Logger::default_level, which is 'info'.
 */
LogLevel log_level_from_env(std::string const& level_env)
{
    return log_level_from_env(level_env, Logger::default_level());
}

//---------------------------------------------------------------------------//
/*!
 * Get the log level from an environment variable.
 */
LogLevel log_level_from_env(std::string const& level_env, LogLevel default_lev)
{
    // Search for the provided environment variable to set the default
    // logging level using the `to_cstring` function in LoggerTypes.
    std::string const& env_value = celeritas::getenv(level_env);
    if (env_value.empty())
    {
        return default_lev;
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
 * The result prints to stderr (buffered through \c std::clog ) only on one
 * processor in the world communicator group.
 * Since it's for messages that should be printed once across all processes,
 * and usually during setup when no local threads are printing, it is not
 * mutexed.
 * This function can be useful when resetting a test harness.
 */
Logger make_default_world_logger()
{
    LogHandler handler;
    if (celeritas::comm_world().rank() == 0)
    {
        handler = StreamLogHandler{std::clog};
    }
    return Logger::from_handle_env(std::move(handler), "CELER_LOG");
}

//---------------------------------------------------------------------------//
/*!
 * Create a default logger using the local communicator.
 *
 * If MPI is enabled, this will prepend the local process index to the message.
 * Because multiple threads and processes can print log messages at the same
 * time, this log output uses a mutex to synchronize output, and prints to
 * buffered stderr through \c std::clog .
 */
Logger make_default_self_logger()
{
    LogHandler handler = LogHandler{MutexedStreamLogHandler{std::clog}};
    if (ScopedMpiInit::status() != ScopedMpiInit::Status::disabled)
    {
        handler = LocalHandler{celeritas::comm_world()};
    }

    return Logger::from_handle_env(std::move(handler), "CELER_LOG_LOCAL");
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
