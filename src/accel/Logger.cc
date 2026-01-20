//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <string>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4Version.hh>
#include <G4ios.hh>

#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/sys/Environment.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantLogger.hh"
#include "accel/detail/LoggerImpl.hh"

namespace celeritas
{
namespace
{

//---------------------------------------------------------------------------//
//! Always write the output, and do not tag thread IDs.
void write_serial(LogProvenance prov, LogLevel lev, std::string msg)
{
    G4cerr << detail::ColorfulLogMessage{prov, lev, msg} << std::endl;
}

//---------------------------------------------------------------------------//
//! Tag a singular output with worker/master: should usually be master
void write_mt_world(LogProvenance prov, LogLevel lev, std::string msg)
{
    if (G4Threading::G4GetThreadId() > 0)
    {
        // Most "CELER_LOG" messages should be during setup, not on a worker,
        // so this should rarely return
        return;
    }

    auto& cerr = G4cerr;
    cerr << color_code('W')
         << (G4Threading::IsMasterThread() ? "[M] " : "[W] ")
         << detail::ColorfulLogMessage{prov, lev, msg} << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Manually create a logger that should only print once in MT or MPI.
 *
 * A given world log message should only print once per execution: on a single
 * process (if using MPI) and a single thread (if using MT).
 * To provide clarity for tasking/MT Geant4 models, this will print whether
 * it's running from a manager `[M]` or worker `[W]` thread if it's a
 * multithreaded app.
 *
 * The \c CELER_LOG_ALL_LOCAL environment variable allows *all* CELER_LOG
 * invocations (on all worker threads) to be written for debugging.
 *
 * In the \c main of your application's executable, set the "process-global"
 * logger:
 * \code
    celeritas::world_logger() = celeritas::MakeMTWorldLogger(*run_manager);
   \endcode
 */
Logger MakeMTWorldLogger(G4RunManager const& runman)
{
    // Assuming the user activates this logger, avoid redirecting future
    // Geant4 messages to avoid recursion
    ScopedGeantLogger::enabled(false);

    LogHandler handle{write_serial};
    if (G4Threading::IsMultithreadedApplication())
    {
        if (getenv_flag("CELER_LOG_ALL_LOCAL", false).value)
        {
            // Every thread lets you know it's being called
            handle = detail::MtSelfWriter{get_geant_num_threads(runman)};
        }
        else
        {
            // Only master and the first worker write
            handle = write_mt_world;
        }
    }
    return Logger{std::move(handle),
                  getenv_loglevel("CELER_LOG", LogLevel::status)};
}

//---------------------------------------------------------------------------//
/*!
 * Manually create a G4MT-friendly logger for event-specific info.
 *
 * This logger redirects Celeritas messages through Geant4.
 * It logger writes the current thread (and maximum number of threads) in
 * each output message, and sends each message through the thread-local \c
 * G4cerr. It should be used for information about a current track or event,
 * specific to the current thread.
 *
 * In the \c main of your application's executable, set the "process-local"
 * logger:
 * \code
    celeritas::self_logger() = celeritas::MakeMTSelfLogger(*run_manager);
   \endcode
 */
Logger MakeMTSelfLogger(G4RunManager const& runman)
{
    // Assuming the user activates this logger, avoid redirecting future
    // Geant4 messages to avoid recursion
    ScopedGeantLogger::enabled(false);

    LogHandler handle{write_serial};
    if (G4Threading::IsMultithreadedApplication())
    {
        handle = detail::MtSelfWriter{get_geant_num_threads(runman)};
    }
    return Logger{std::move(handle),
                  getenv_loglevel("CELER_LOG_LOCAL", LogLevel::warning)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
