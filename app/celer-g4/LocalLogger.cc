//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/LocalLogger.cc
//---------------------------------------------------------------------------//
#include "LocalLogger.hh"

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
void LocalLogger::operator()(Provenance prov, LogLevel lev, std::string msg)
{
    int local_thread = G4Threading::G4GetThreadId();
    std::clog << color_code('W') << '[';
    if (local_thread >= 0)
    {
        std::clog << local_thread + 1;
    }
    else
    {
        std::clog << 'M';
    }
    std::clog << '/' << num_threads_ << "] " << color_code(' ');

    if (lev == LogLevel::debug || lev >= LogLevel::warning)
    {
        // Output problem line/file for debugging or high level
        std::clog << color_code('x') << prov.file;
        if (prov.line)
            std::clog << ':' << prov.line;
        std::clog << color_code(' ') << ": ";
    }

    std::clog << to_color_code(lev) << to_cstring(lev) << ": "
              << color_code(' ') << msg << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
