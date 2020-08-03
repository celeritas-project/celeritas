//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ColorUtils.cc
//---------------------------------------------------------------------------//
#include "ColorUtils.hh"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>

namespace celeritas
{
namespace
{
bool determine_use_color(FILE* stream)
{
    int fn = fileno(stream);
    if (!isatty(fn))
        return false;

    const char* term = std::getenv("TERM");
    if (!term)
        return false;

    if (strstr("xterm", term) == 0)
        return true;

    // Other checks here? ...

    return false;
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether colors are enabled (currently read-only)
 */
bool use_color()
{
    const static bool result = determine_use_color(stdout);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get an ANSI color codes if colors are enabled.
 *
 *  - [y]ellow
 *  - [g]reen
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
