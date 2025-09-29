//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
namespace
{
//---------------------------------------------------------------------------//
// Get a default color based on the terminal settings
bool default_term_color()
{
#ifndef _WIN32
    if (isatty(fileno(stderr)))
    {
        // This stream is a user-facing terminal
        return true;
    }
#endif
    if (char const* term_str = std::getenv("TERM"))
    {
        if (std::string{term_str}.find("xterm") != std::string::npos)
        {
            // 'xterm' is in the TERM type, so assume it uses colors
            return true;
        }
    }
    return false;
}

// Get a default color based on the terminal settings *or* gtest override
bool default_gtest_or_term_color()
{
    // Don't use celeritas getenv to check gtest variable, to avoid
    // adding it to the list of exposed variables if unused
    if (char const* color_cstr = std::getenv("GTEST_COLOR"))
    {
        // Since it's used, add it to the environment and use the flag logic to
        // process its value
        return celeritas::getenv_flag("GTEST_COLOR", default_term_color).value;
    }
    return default_term_color();
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether colors are enabled (currently read-only).
 */
bool use_color()
{
    static bool const result
        = celeritas::getenv_flag("CELER_COLOR", default_gtest_or_term_color).value;
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
        default:
            return "\033[0m";
    }

    // Unknown color code: ignore
    return "";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
