//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Logger.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Macros.hh"

#include "LoggerTypes.hh"

#include "detail/LoggerMessage.hh"  // IWYU pragma: export
#if CELER_DEVICE_COMPILE
#    include "detail/NullLoggerMessage.hh"  // IWYU pragma: export
#endif

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
//! Inject the source code provenance (current file and line)
#define CELER_CODE_PROVENANCE  \
    ::celeritas::LogProvenance \
    {                          \
        __FILE__, __LINE__     \
    }

/*!
 * \def CELER_LOG
 *
 * Return a LogMessage object for streaming into at the given level. The
 * regular \c CELER_LOG call is for code paths that happen uniformly in
 * parallel, approximately the same message from every thread and task.
 *
 * The logger will only format and print messages. It is not responsible
 * for cleaning up the state or exiting an app.
 *
 * \code
 CELER_LOG(debug) << "Don't print this in general";
 CELER_LOG(warning) << "You may want to reconsider your life choices";
 CELER_LOG(critical) << "Caught a fatal exception: " << e.what();
 * \endcode
 */
#define CELER_LOG(LEVEL)                               \
    ::celeritas::world_logger()(CELER_CODE_PROVENANCE, \
                                ::celeritas::LogLevel::LEVEL)

/*!
 * \def CELER_LOG_LOCAL
 *
 * Like \c CELER_LOG but for code paths that may only happen on a single
 * process or thread. Use sparingly because this can be very verbose. This
 * should be used for error messages from an event or
 * track at runtime.
 */
#define CELER_LOG_LOCAL(LEVEL)                        \
    ::celeritas::self_logger()(CELER_CODE_PROVENANCE, \
                               ::celeritas::LogLevel::LEVEL)

// Allow CELER_LOG to be present (but ignored) in device code
#if CELER_DEVICE_COMPILE
#    undef CELER_LOG
#    define CELER_LOG(LEVEL) ::celeritas::detail::NullLoggerMessage()
#    undef CELER_LOG_LOCAL
#    define CELER_LOG_LOCAL(LEVEL) ::celeritas::detail::NullLoggerMessage()
#endif

namespace celeritas
{
class MpiCommunicator;

//---------------------------------------------------------------------------//
/*!
 * Create a log message to be printed based on output/verbosity settings.
 *
 * This should generally be called by the \c world_logger and \c
 * self_logger functions below. The call \c operator() returns an object that
 * should be streamed into in order to create a log message.
 *
 * This object \em is assignable, so to replace the default log handler with a
 * different one, you can call \code
   world_logger = Logger(my_handler);
 * \endcode
 *
 * When using with MPI, the \c world_logger global objects are different on
 * each process: rank 0 will have a handler that outputs to screen, and the
 * other ranks will have a "null" handler that suppresses all log output.
 *
 * \todo Replace the back-end with \c spdlog to reduce maintenance
 * burden and improve flexibility.
 */
class Logger
{
  public:
    //!@{
    //! \name Type aliases
    using Message = detail::LoggerMessage;
    //!@}

  public:
    //! Get the default log level
    static constexpr LogLevel default_level() { return LogLevel::status; }

    // Construct a null logger
    Logger() = default;

    // Construct a logger with handle and default level
    explicit Logger(LogHandler&& handle);

    // Construct a logger with handle and level
    Logger(LogHandler&& handle, LogLevel min_lev);

    // Create a log message that flushes its contents when it destructs
    inline Message operator()(LogProvenance&& prov, LogLevel lev) const;

    //! Set the minimum logging verbosity
    void level(LogLevel lev) { min_level_ = lev; }

    //! Get the current logging verbosity
    LogLevel level() const { return min_level_; }

    //! Access the log handle
    LogHandler const& handle() const { return handle_; }

    //! Set the log handle (null disables logger)
    void handle(LogHandler&& handle) { handle_ = std::move(handle); }

  private:
    LogHandler handle_;
    LogLevel min_level_{default_level()};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create a logger that flushes its contents when it destructs.
 *
 * It's assumed that log messages will be relatively unlikely (and expensive
 * anyway), so we mark as \c CELER_UNLIKELY to optimize for the no-logging
 * case.
 */
auto Logger::operator()(LogProvenance&& prov, LogLevel lev) const -> Message
{
    LogHandler const* handle = nullptr;
    if (CELER_UNLIKELY(handle_ && lev >= min_level_))
    {
        handle = &handle_;
    }
    return Message{handle, std::move(prov), lev};
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Parallel logger (print only on "main" process)
Logger& world_logger();

// Serial logger (print on *every* process)
Logger& self_logger();

//---------------------------------------------------------------------------//
}  // namespace celeritas
