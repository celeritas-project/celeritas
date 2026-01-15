//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LogHandlers.cc
//---------------------------------------------------------------------------//
#include "LogHandlers.hh"

#include <mutex>
#include <ostream>
#include <sstream>

#include "corecel/sys/MpiCommunicator.hh"

#include "ColorUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple log handler: write with colors to an ostream reference.
 */
void StreamLogHandler::operator()(LogProvenance prov,
                                  LogLevel lev,
                                  std::string msg) const
{
    if (lev < LogLevel::status || lev >= LogLevel::warning)
    {
        // Output problem line/file for debugging or high level
        os_ << color_code('x') << prov.file;
        if (prov.line)
        {
            os_ << ':' << prov.line;
        }
        os_ << color_code(' ') << ": ";
    }

    os_ << to_ansi_color(lev) << to_cstring(lev) << ": " << color_code(' ');
    os_ << std::move(msg) << std::endl;
}

//---------------------------------------------------------------------------//
/*!
 * Simple log handler: write with colors to an ostream reference.
 */
void MutexedStreamLogHandler::operator()(LogProvenance prov,
                                         LogLevel lev,
                                         std::string msg) const
{
    std::ostringstream temp_os;
    StreamLogHandler{temp_os}(prov, lev, std::move(msg));

    // Lock after building the message
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> scoped_lock{log_mutex};
    os_ << std::move(temp_os).str() << std::flush;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with long-lived reference to a stream.
 */
LocalMpiHandler::LocalMpiHandler(std::ostream& os, MpiCommunicator const& comm)
    : os_{os}, rank_(comm.rank())
{
}

// Write with processor ID
void LocalMpiHandler::operator()(LogProvenance prov,
                                 LogLevel lev,
                                 std::string msg) const
{
    // Buffer to reduce I/O contention in MPI runner
    std::ostringstream os;
    os << color_code('W') << "rank " << rank_ << ": ";
    StreamLogHandler{os}(prov, lev, msg);

    os_ << std::move(os).str() << std::flush;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
