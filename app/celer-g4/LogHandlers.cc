//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/LogHandlers.cc
//---------------------------------------------------------------------------//
#include "LogHandlers.hh"

#include <mutex>
#include <G4Threading.hh>

#include "corecel/io/ColorUtils.hh"
#include "corecel/io/LogHandlers.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "geocel/GeantUtils.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
//! Print the message and flush atomically to stderr
void write_msg(std::ostringstream&& os,
               LogProvenance const& prov,
               LogLevel const& lev,
               std::string const& msg)
{
    // Write main message
    StreamLogHandler{os}(prov, lev, msg);

    // Lock after building the message while writing
    static std::mutex log_mutex;
    std::lock_guard scoped_lock{log_mutex};
    std::clog << std::move(os).str() << std::flush;
}

//---------------------------------------------------------------------------//
/*!
 * Write a log message when only a single thread.
 */
void handle_serial(LogProvenance prov, LogLevel lev, std::string msg)
{
    return write_msg(std::ostringstream{}, prov, lev, msg);
}

//---------------------------------------------------------------------------//
//! Tag a singular output with worker/main: should usually be main
void handle_mt_world(LogProvenance prov, LogLevel lev, std::string msg)
{
    if (G4Threading::G4GetThreadId() > 0)
    {
        // Most "CELER_LOG" messages should be during setup, not on a worker,
        // so this should rarely return
        return;
    }

    std::ostringstream os;
    os << color_code('W') << (G4Threading::IsMasterThread() ? "[M] " : "[W] ")
       << color_code(' ');

    return write_msg(std::move(os), prov, lev, msg);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from number of threads.
 */
SelfLogHandler::SelfLogHandler(int num_threads) : num_threads_(num_threads)
{
    CELER_EXPECT(num_threads_ > 0);
    if (auto& comm = comm_world())
    {
        rank_ = comm.rank();
        size_ = comm.size();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Write a log message.
 */
void SelfLogHandler::operator()(LogProvenance prov,
                                LogLevel lev,
                                std::string msg)
{
    // Write preamble to a buffer first
    std::ostringstream os;

    int local_thread = G4Threading::G4GetThreadId();
    os << color_code('W') << '[';
    if (size_ > 0)
    {
        // Print MPI rank
        os << rank_ + 1 << '/' << size_ << ':';
    }
    if (local_thread >= 0)
    {
        os << local_thread + 1;
    }
    else
    {
        os << 'M';
    }
    os << '/' << num_threads_ << "] ";

    return write_msg(std::move(os), prov, lev, msg);
}

//---------------------------------------------------------------------------//
/*!
 * Create a handler for "everyone logs the same" messages.
 *
 * - If MPI and not the head process, return no handler to silence logging.
 * - If not using Geant4 MT, don't annotate threads.
 * - If using MT and CELER_LOG_ALL_LOCAL is set, print the thread-annotated
 *   global messages from every thread.
 * - Otherwise, only a single thread logs. If it's a worker thread logging, it
 *   gets a W prefix, else M.
 */
LogHandler make_world_handler()
{
    if (auto& comm = comm_world())
    {
        if (comm.rank() != 0)
        {
            // Do not log from any process but the first
            return {};
        }
    }
    if (!G4Threading::IsMultithreadedApplication())
    {
        return handle_serial;
    }
    if (getenv_flag("CELER_LOG_ALL_LOCAL", false).value)
    {
        // Every thread lets you know it's being called
        return SelfLogHandler{get_geant_num_threads()};
    }

    // Only the main thread (and a single worker if MT) write
    return handle_mt_world;
}

LogHandler make_self_handler(int num_threads)
{
    if (G4Threading::IsMultithreadedApplication())
    {
        return SelfLogHandler{num_threads};
    }
    return handle_serial;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
