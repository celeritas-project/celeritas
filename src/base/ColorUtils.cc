//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ColorUtils.cc
//---------------------------------------------------------------------------//
#include "ColorUtils.hh"

#include <cstdio>
#include <unistd.h>
#include "comm/Environment.hh"

namespace celeritas
{
namespace
{
bool determine_use_color(FILE* stream)
{
    const std::string& color_str = celeritas::getenv("GTEST_COLOR");
    if (color_str == "0")
    {
        // GTEST_COLOR explicitly disables color
        return false;
    }
    else if (!color_str.empty())
    {
        // GTEST_COLOR explicitly enables color
        return true;
    }

    if (isatty(fileno(stream)))
    {
        // Given stream says it's a "terminal" i.e. user-facing
        const std::string& term_str = celeritas::getenv("TERM");
        if (term_str.find("xterm") != std::string::npos)
        {
            // 'xterm' is in the TERM type, so assume it uses colors
            return true;
        }
    }

    return false;
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether colors are enabled (currently read-only).
 */
bool use_color()
{
    const static bool result = determine_use_color(stderr);
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
const char* color_code(char abbrev)
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
} // namespace celeritas
