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

#include "corecel/Assert.hh"
#include "corecel/sys/Environment.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Get a default color based on the terminal settings
bool default_term_color()
{
#ifndef _WIN32
    // See if stderr is a user-facing terminal
    return isatty(fileno(stderr));
#endif
    // Fall back to checking environment variable
    if (auto term_str = celeritas::getenv("TERM"); !term_str.empty())
    {
        if (term_str.find("xterm") != std::string::npos)
        {
            // 'xterm' is in the TERM type, so assume it uses colors
            return true;
        }
    }
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Get the preferred environment variable to use for color override.
 */
char const* color_env_var()
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
 * Whether colors are enabled by the environment.
 *
 * The \c NO_COLOR environment variable, if set to a non-empty value, disables
 * color output. If either of the \c CELER_COLOR or \c GTEST_COLOR variables is
 * set, that value will be used. Failing that, the default is true if \c stderr
 * is a tty. The result of this value is used by \c ansi_color .
 */
bool use_color()
{
    static bool const result = [] {
        if (auto term_str = celeritas::getenv("NO_COLOR"); !term_str.empty())
        {
            // See https://no-color.org
            return false;
        }

        // Check one environment variable and fall back to terminal color
        auto flag = getenv_flag_lazy(color_env_var(), default_term_color);
        return flag.value;
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
 *  - \c null reset color
 *  - [ ] reset color
 */
char const* ansi_color(char abbrev)
{
    if (!use_color())
        return "";

    switch (abbrev)
    {
        case '\0':
            [[fallthrough]];
        case ' ':
            return "\033[0m";
        case 'r':
            return "\033[31m";
        case 'g':
            return "\033[32m";
        case 'y':
            return "\033[33m";
        case 'b':
            return "\033[34m";
        case 'R':
            return "\033[1;31m";
        case 'G':
            return "\033[1;32m";
        case 'B':
            return "\033[1;34m";
        case 'W':
            return "\033[1;37m";
        case 'x':
            return "\033[2;37m";
        default:
            return "\033[0m";
    }

    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
