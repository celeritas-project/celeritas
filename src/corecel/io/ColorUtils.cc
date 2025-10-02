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
    // See if stderr is a user-facing terminal
    return isatty(fileno(stderr));
#endif
    // Fall back to checking environment variable
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

//---------------------------------------------------------------------------//
/*!
 * Get a default color based on the terminal/env settings.
 */
char const* default_color_env_str()
{
    auto hasenv = [](char const* key) { return std::getenv(key) != nullptr; };
    static char const celer_env[] = "CELER_COLOR";
    static char const gtest_env[] = "GTEST_COLOR";

    if (hasenv(celer_env) || !hasenv(gtest_env))
        return celer_env;
    return gtest_env;
}

}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether colors are enabled (currently read-only).
 *
 * This checks the \c CELER_COLOR environment variable, or the \c
 * GTEST_COLOR variable (if it is defined and CELER_COLOR is not), and defaults
 * based on the terminal settings of \c stderr.
 */
bool use_color()
{
    static bool const result
        = celeritas::getenv_flag(default_color_env_str(), default_term_color())
              .value;
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
