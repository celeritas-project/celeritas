//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LoggerTypes.hh
//! \brief Type definitions for logging utilities
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Enumeration for how important a log message is.
 */
enum class LogLevel
{
    debug,  //!< Debugging messages
    diagnostic,  //!< Diagnostics about current program execution
    status,  //!< Program execution status (what stage is beginning)
    info,  //!< Important informational messages
    warning,  //!< Warnings about unusual events
    error,  //!< Something went wrong, but execution can continue
    critical,  //!< Something went terribly wrong, should probably abort
    size_  //!< Sentinel value for looping over log levels
};

//---------------------------------------------------------------------------//
// Get the plain text equivalent of the log level above
char const* to_cstring(LogLevel);

//---------------------------------------------------------------------------//
// Get an ANSI color code appropriate to each log level
char const* to_color_code(LogLevel);

//---------------------------------------------------------------------------//
//! Stand-in for a more complex class for the "provenance" of data
struct Provenance
{
    std::string file;  //!< Originating file
    int line = 0;  //!< Line number
};

//! Type for handling a log message
using LogHandler = std::function<void(Provenance, LogLevel, std::string)>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
