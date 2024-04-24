//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Assert.cc
//---------------------------------------------------------------------------//
#include "Assert.hh"

#include "Macros.hh"

#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

#include <sstream>
#include <utility>

#include "io/ColorUtils.hh"
#include "sys/Environment.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Construct a debug assertion message for printing.
 */
std::string build_debug_error_msg(DebugErrorDetails const& d)
{
    std::ostringstream msg;
    // clang-format off
    msg << color_code('W') << d.file << ':' << d.line << ':'
        << color_code(' ') << "\nceleritas: "
        << color_code('R') << to_cstring(d.which);
    // clang-format on
    if (d.which != DebugErrorType::unreachable)
    {
        msg << ": " << color_code('x') << d.condition;
    }
    msg << color_code(' ');
    return msg.str();
}

//---------------------------------------------------------------------------//
/*!
 * Construct a runtime assertion message for printing.
 */
std::string build_runtime_error_msg(RuntimeErrorDetails const& d)
{
    static bool const verbose_message = [] {
#if CELERITAS_DEBUG
        // Always verbose if debug flags are enabled
        return true;
#else
        // Verbose if the CELER_LOG environment variable is defined
        return !celeritas::getenv("CELER_LOG").empty();
#endif
    }();

    std::ostringstream msg;

    msg << "celeritas: " << color_code('R')
        << (d.which.empty() ? "unknown" : d.which)
        << " error: " << color_code(' ');
    if (d.which == "not configured")
    {
        msg << "required dependency is disabled in this build: ";
    }
    else if (d.which == "not implemented")
    {
        msg << "feature is not yet implemented: ";
    }
    msg << d.what;

    if (verbose_message || d.what.empty() || d.which != "runtime")
    {
        msg << '\n'
            << color_code('W') << (d.file.empty() ? "unknown source" : d.file);
        if (d.line)
        {
            msg << ':' << d.line;
        }

        msg << ':' << color_code(' ');
        if (!d.condition.empty())
        {
            msg << " '" << d.condition << '\'' << " failed";
        }
        else
        {
            msg << " " << d.which;
        }
    }

    return msg.str();
}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get a human-readable string describing a debug error.
 */
char const* to_cstring(DebugErrorType which)
{
    switch (which)
    {
        case DebugErrorType::precondition:
            return "precondition failed";
        case DebugErrorType::internal:
            return "internal assertion failed";
        case DebugErrorType::unreachable:
            return "unreachable code point";
        case DebugErrorType::postcondition:
            return "postcondition failed";
        case DebugErrorType::assumption:
            return "assumption failed";
    }
    return "";
}

//---------------------------------------------------------------------------//
/*!
 * Get an MPI error string.
 */
std::string mpi_error_to_string(int errorcode)
{
#if CELERITAS_USE_MPI
    std::string error_string;
    error_string.resize(MPI_MAX_ERROR_STRING);
    int length = 0;
    MPI_Error_string(errorcode, &error_string.front(), &length);
    error_string.resize(length);
    return error_string;
#else
    CELER_DISCARD(errorcode);
    CELER_NOT_CONFIGURED("MPI");
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Construct a debug exception from detailed attributes.
 */
DebugError::DebugError(DebugErrorDetails&& d)
    : std::logic_error(build_debug_error_msg(d)), details_(std::move(d))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a runtime error from detailed descriptions.
 */
RuntimeError::RuntimeError(RuntimeErrorDetails&& d)
    : std::runtime_error(build_runtime_error_msg(d)), details_(std::move(d))
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
