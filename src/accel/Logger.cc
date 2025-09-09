//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <algorithm>
#include <mutex>
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

namespace celeritas
{
namespace
{

//---------------------------------------------------------------------------//
//! Write a colorful filename to a mysterious Geant4 streamable
template<class T>
void write_log(T& os,
               LogProvenance const& prov,
               LogLevel lev,
               std::string const& msg)
{
    os << to_color_code(lev) << to_cstring(lev);
    if (!prov.file.empty())
    {
        os << color_code('x') << "@";

        // Write the file name up to the last directory component
        auto last_slash = std::find(prov.file.rbegin(), prov.file.rend(), '/');
        if (!prov.file.empty() && last_slash == prov.file.rend())
        {
            --last_slash;
        }

        // Output problem line/file for debugging or high level
        os << std::string(last_slash.base(), prov.file.end());
        if (prov.line)
        {
            os << ':' << prov.line;
        }
    }
    os << color_code(' ') << ": " << msg << std::endl;
}

//---------------------------------------------------------------------------//
//! Write the thread ID on output
class MtSelfWriter
{
  public:
    explicit MtSelfWriter(int num_threads);
    void operator()(LogProvenance prov, LogLevel lev, std::string msg);

  private:
    int num_threads_;
};

MtSelfWriter::MtSelfWriter(int num_threads) : num_threads_(num_threads)
{
    CELER_EXPECT(num_threads_ >= 0);
}

void MtSelfWriter::operator()(LogProvenance prov, LogLevel lev, std::string msg)
{
    auto& cerr = G4cerr;

    int local_thread = G4Threading::G4GetThreadId();
    if (local_thread >= 0)
    {
        // Logging from a worker thread
        if (CELER_UNLIKELY(local_thread >= num_threads_))
        {
            // In tasking or potentially other contexts, the max thread might
            // not be known. Update it here for better output.
            static std::mutex thread_update_mutex;
            std::lock_guard scoped_lock{thread_update_mutex};
            num_threads_ = std::max(local_thread + 1, num_threads_);
        }

        cerr << color_code('W') << '[' << local_thread + 1 << '/'
             << num_threads_ << "] ";
    }
    else
    {
        // Logging "local" message from the master thread!
        cerr << color_code('W') << "[M!] ";
    }
    cerr << color_code(' ');

    return write_log(cerr, prov, lev, msg);
}

//---------------------------------------------------------------------------//
//! Always write the output, and do not tag thread IDs.
void write_serial(LogProvenance prov, LogLevel lev, std::string msg)
{
    return write_log(G4cerr, prov, lev, msg);
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
         << color_code(' ');

    return write_log(cerr, prov, lev, msg);
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
            handle = MtSelfWriter{get_geant_num_threads(runman)};
        }
        else
        {
            // Only master and the first worker write
            handle = write_mt_world;
        }
    }
    return Logger::from_handle_env(std::move(handle), "CELER_LOG");
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
        handle = MtSelfWriter{get_geant_num_threads(runman)};
    }
    return Logger::from_handle_env(std::move(handle), "CELER_LOG_LOCAL");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
