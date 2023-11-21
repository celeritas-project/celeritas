//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LoggerTypes.cc
//---------------------------------------------------------------------------//
#include "LoggerTypes.hh"

#include "corecel/io/EnumStringMapper.hh"

#include "ColorUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the plain text equivalent of the LogLevel enum.
 */
char const* to_cstring(LogLevel lev)
{
    static EnumStringMapper<LogLevel> const to_cstring_impl{
        "debug",
        "diagnostic",
        "status",
        "info",
        "warning",
        "error",
        "critical",
    };
    return to_cstring_impl(lev);
}

//---------------------------------------------------------------------------//
/*!
 * Get an ANSI color code appropriate to each log level.
 */
char const* to_color_code(LogLevel lev)
{
    // clang-format off
    char c = ' ';
    switch (lev)
    {
        case LogLevel::debug:      c = 'x'; break;
        case LogLevel::diagnostic: c = 'x'; break;
        case LogLevel::status:     c = 'b'; break;
        case LogLevel::info:       c = 'g'; break;
        case LogLevel::warning:    c = 'y'; break;
        case LogLevel::error:      c = 'r'; break;
        case LogLevel::critical:   c = 'R'; break;
        case LogLevel::size_: CELER_ASSERT_UNREACHABLE();
    };
    // clang-format on

    return color_code(c);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
