//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Assert.cc
//---------------------------------------------------------------------------//
#include "Assert.hh"

#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

#include <sstream>

#include "io/ColorUtils.hh"
#include "sys/Environment.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
bool determine_verbose_message()
{
#if CELERITAS_DEBUG
    // Always verbose if debug flags are enabled
    return true;
#else
    // Verbose if the CELER_LOG environment variable is defined
    return !celeritas::getenv("CELER_LOG").empty();
#endif
}

//---------------------------------------------------------------------------//
const char* to_cstring(DebugErrorType which)
{
    switch (which)
    {
        case DebugErrorType::precondition:
            return "precondition failed";
        case DebugErrorType::internal:
            return "internal assertion failed";
        case DebugErrorType::unreachable:
            return "unreachable code point";
        case DebugErrorType::unconfigured:
            return "required dependency is disabled in this build";
        case DebugErrorType::unimplemented:
            return "feature is not yet implemented";
        case DebugErrorType::postcondition:
            return "postcondition failed";
    }
    return "";
}
} // namespace

//---------------------------------------------------------------------------//
//!@{
//! Delegating constructor
DebugError::DebugError(const char* msg) : std::logic_error(msg) {}
DebugError::DebugError(const std::string& msg) : std::logic_error(msg) {}
RuntimeError::RuntimeError(const char* msg) : std::runtime_error(msg) {}
RuntimeError::RuntimeError(const std::string& msg) : std::runtime_error(msg) {}
//!@}

//---------------------------------------------------------------------------//
/*!
 * Construct a debug assertion message and throw.
 */
[[noreturn]] void throw_debug_error(DebugErrorType which,
                                    const char*    condition,
                                    const char*    file,
                                    int            line)
{
    std::ostringstream msg;
    // clang-format off
    msg << color_code('W') << file << ':' << line << ':'
        << color_code(' ') << "\nceleritas: "
        << color_code('R') << to_cstring(which);
    // clang-format on
    if (which != DebugErrorType::unreachable)
    {
        msg << ": " << color_code('x') << condition;
    }
    msg << color_code(' ');
    throw DebugError(msg.str());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a message and throw an error from a runtime CUDA/HIP failure.
 */
[[noreturn]] void throw_device_call_error(const char* error_string,
                                          const char* code,
                                          const char* file,
                                          int         line)
{
    std::ostringstream msg;
    // clang-format off
    msg << color_code('W') << file << ':' << line << ':'
        << color_code(' ') << "\nceleritas: "
        << color_code('R') << (CELERITAS_USE_CUDA ? "cuda" : "device") << " error: "
        << color_code(' ') << error_string << "\n    "
        << color_code('x') << code
        << color_code(' ');
    // clang-format on
    throw RuntimeError(msg.str());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a message and throw an error from a runtime MPI failure.
 */
[[noreturn]] void throw_mpi_call_error(CELER_MAYBE_UNUSED int errorcode,
                                       const char*            code,
                                       const char*            file,
                                       int                    line)
{
    std::string error_string;
#if CELERITAS_USE_MPI
    {
        error_string.resize(MPI_MAX_ERROR_STRING);
        int length = 0;
        MPI_Error_string(errorcode, &error_string.front(), &length);
        error_string.resize(length);
    }
#endif
    std::ostringstream msg;
    // clang-format off
    msg << color_code('W') << file << ':' << line << ':'
        << color_code(' ') << "\nceleritas: "
        << color_code('R') << "mpi error: "
        << color_code(' ') << error_string << "\n    "
        << color_code('x') << code
        << color_code(' ');
    // clang-format on
    throw RuntimeError(msg.str());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a runtime assertion message.
 */
[[noreturn]] void throw_runtime_error(std::string detail,
                                      const char* condition,
                                      const char* file,
                                      int         line)
{
    static const bool verbose_message = determine_verbose_message();

    std::ostringstream msg;

    if (verbose_message)
    {
        msg << color_code('W') << file << ':' << line << ':' << color_code(' ')
            << '\n';
    }

    msg << "celeritas: " << color_code('R') << "runtime error: ";
    if (verbose_message || detail.empty())
    {
        msg << color_code('x') << condition << color_code(' ') << " failed";
        if (!detail.empty())
            msg << ":\n    ";
    }
    else
    {
        msg << color_code(' ');
    }
    msg << detail;

    throw RuntimeError(msg.str());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
