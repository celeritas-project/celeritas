//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/LocalLogger.cc
//---------------------------------------------------------------------------//
#include "LocalLogger.hh"

#include <mutex>
#include <G4Threading.hh>

#include "corecel/io/ColorUtils.hh"

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

    if (lev == LogLevel::debug || lev >= LogLevel::warning)
    {
        // Output problem line/file for debugging or high level
        os << color_code('x') << prov.file;
        if (prov.line)
            os << ':' << prov.line;
        os << color_code(' ') << ": ";
    }
    os << to_color_code(lev) << to_cstring(lev) << ": " << color_code(' ');

    {
        // Write buffered content and message with a mutex, then flush
        static std::mutex clog_mutex;
        std::lock_guard scoped_lock{clog_mutex};
        std::clog << os.str() << msg << std::endl;
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
