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

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
//! Finalize the message, including
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
    std::clog << std::move(os).str();
}

//---------------------------------------------------------------------------//
}  // namespace

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
 * Write a "world"-level log message.
 */
void handle_world_log(LogProvenance prov, LogLevel lev, std::string msg)
{
    // Write preamble to a buffer first
    std::ostringstream os;

    os << color_code('W') << "[W] " << color_code(' ');

    return write_msg(std::move(os), prov, lev, msg);
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
