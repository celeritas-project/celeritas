//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ColorUtils.cc
//---------------------------------------------------------------------------//
#include "ColorUtils.hh"

#include <cstdio>
#include <cstdlib>
#include <string>
#ifndef _WIN32
#    include <unistd.h>
#endif

#include "corecel/sys/Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Whether colors are enabled (currently read-only).
 */
bool use_color()
{
    const static bool result = [] {
        [[maybe_unused]] FILE* stream = stderr;
        std::string color_str = celeritas::getenv("CELER_COLOR");
        if (color_str.empty())
        {
            // Don't use celeritas getenv to check gtest variable, to avoid
            // adding it to the list of exposed variables
            if (const char* color_cstr = std::getenv("GTEST_COLOR"))
            {
                color_str = std::string(color_cstr);
            }
        }
        if (color_str == "0")
        {
            // Color is explicitly disabled
            return false;
        }
        if (!color_str.empty())
        {
            // Color is explicitly enabled
            return true;
        }
#ifndef _WIN32
        if (!isatty(fileno(stream)))
        {
            // This stream is not a user-facing terminal
            return false;
        }
#endif
        if (const char* term_str = std::getenv("TERM"))
        {
            if (std::string{term_str}.find("xterm") != std::string::npos)
            {
                // 'xterm' is in the TERM type, so assume it uses colors
                return true;
            }
        }

        return false;
    }();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get an ANSI color codes if colors are enabled.
 *
 *  - [b]lue
 *  - [g]reen
 *  - [y]ellow
 *  - [r]ed
 *  - [x] gray
 *  - [R]ed bold
 *  - [W]hite bold
 *  - [ ] default (reset color)
 */
char const* color_code(char abbrev)
{
    if (!use_color())
        return "";

    switch (abbrev)
    {
        case 'g':
            return "\033[32m";
        case 'b':
            return "\033[34m";
        case 'r':
            return "\033[31m";
        case 'x':
            return "\033[37;2m";
        case 'y':
            return "\033[33m";
        case 'R':
            return "\033[31;1m";
        case 'W':
            return "\033[37;1m";
        case ' ':
            return "\033[0m";
    }

    // Unknown color code: ignore
    return "";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
