//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LoggerTypes.cc
//---------------------------------------------------------------------------//
#include "LoggerTypes.hh"

#include <algorithm>
#include <string>

#include "corecel/cont/Range.hh"
#include "corecel/sys/Environment.hh"

#include "ColorUtils.hh"
#include "EnumStringMapper.hh"

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
char const* to_ansi_color(LogLevel lev)
{
    // clang-format off
    char c = ' ';
    switch (lev)
    {
        case LogLevel::debug:     [[fallthrough]];
        case LogLevel::diagnostic: c = 'x'; break;
        case LogLevel::status:     c = 'b'; break;
        case LogLevel::info:       c = 'g'; break;
        case LogLevel::warning:    c = 'y'; break;
        case LogLevel::error:      c = 'r'; break;
        case LogLevel::critical:   c = 'R'; break;
        case LogLevel::size_: CELER_ASSERT_UNREACHABLE();
    };
    // clang-format on

    return ansi_color(c);
}

//---------------------------------------------------------------------------//
/*!
 * Get the log level from an environment variable.
 *
 * \throws RuntimeError if string is invalid
 */
LogLevel getenv_loglevel(std::string const& level_env, LogLevel default_lev)
{
    // Search for the provided environment variable to set the default
    // logging level using the `to_cstring` function in LoggerTypes.
    std::string const& env_value = celeritas::getenv(level_env);
    if (env_value.empty())
    {
        return default_lev;
    }

    auto levels = range(LogLevel::size_);
    auto iter = std::find_if(
        levels.begin(), levels.end(), [&env_value](LogLevel lev) {
            return env_value == to_cstring(lev);
        });
    CELER_VALIDATE(iter != levels.end(),
                   << "invalid log level '" << env_value
                   << "' in environment variable '" << level_env << "'");
    return *iter;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
