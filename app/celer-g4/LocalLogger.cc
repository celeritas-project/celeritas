//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/LocalLogger.cc
//---------------------------------------------------------------------------//
#include "LocalLogger.hh"

#include <mutex>
#include <G4Threading.hh>

#include "corecel/io/ColorUtils.hh"
#include "corecel/io/LogHandlers.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Write a log message.
 */
void LocalLogger::operator()(LogProvenance prov, LogLevel lev, std::string msg)
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
    os << '/' << num_threads_ << "] " << color_code(' ');

    // Write main message
    StreamLogHandler{os}(prov, lev, msg);

    // Lock after building the message
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> scoped_lock{log_mutex};
    std::clog << std::move(os).str();
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
