//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <cstdlib>
#include <string>

#include <string.h>
#include <stdio.h>
#include <unistd.h>

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

bool use_color()
{
    const static bool result = determine_use_color(stdout);
    return result;
}
} // namespace

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Get an ANSI color codes if colors are enabled.
 *
 *  - [y]ellow
 *  - [g]reen
 *  - [r]ed
 *  - [x] gray
 *  - [ ] default.
 */
const char* color_code(char abbrev)
{
    if (!use_color())
        return "";

    switch (abbrev)
    {
        case 'g':
            return "\e[32m";
        case 'r':
            return "\e[31m";
        case 'x':
            return "\e[37;2m";
        case 'y':
            return "\e[33m";
        case ' ':
            // Reset color
            return "\e[0m";
    }

    // Unknown color code: ignore
    return "";
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the "skip" message for the skip macro
 */
const char* skip_cstring()
{
    static const std::string str = std::string(color_code('y'))
                                   + std::string("[   SKIP   ]")
                                   + std::string(color_code('d'));
    return str.c_str();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
