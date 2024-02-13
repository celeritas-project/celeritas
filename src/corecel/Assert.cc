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
 * Return whether to give an extra verbose message.
 */
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
    static bool const verbose_message = determine_verbose_message();

    std::ostringstream msg;

    if (d.which != RuntimeErrorType::validate || verbose_message)
    {
        msg << color_code('W') << d.file;
        if (d.line)
        {
            msg << ':' << d.line;
        }
        msg << ':' << color_code(' ') << '\n';
    }

    msg << "celeritas: " << color_code('R') << to_cstring(d.which)
        << " error: ";
    if (verbose_message || d.what.empty())
    {
        msg << color_code('x') << d.condition << color_code(' ') << " failed";
        if (!d.what.empty())
            msg << ":\n    ";
    }
    else
    {
        msg << color_code(' ');
    }
    msg << d.what;

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
        case DebugErrorType::unconfigured:
            return "required dependency is disabled in this build";
        case DebugErrorType::unimplemented:
            return "feature is not yet implemented";
        case DebugErrorType::postcondition:
            return "postcondition failed";
        case DebugErrorType::assumption:
            return "assumption failed";
    }
    return "";
}

//---------------------------------------------------------------------------//
/*!
 * Get a human-readable string describing a runtime error.
 */
char const* to_cstring(RuntimeErrorType which)
{
    switch (which)
    {
        case RuntimeErrorType::validate:
            return "runtime";
        case RuntimeErrorType::device:
#if CELERITAS_USE_CUDA
            return "CUDA";
#else
            return "device";
#endif
        case RuntimeErrorType::mpi:
            return "MPI";
        case RuntimeErrorType::geant:
            return "Geant4";
        case RuntimeErrorType::root:
            return "ROOT";
    }
    return "";
}

//---------------------------------------------------------------------------//
/*!
 * Construct a debug exception from detailed attributes.
 */
DebugError::DebugError(DebugErrorDetails d)
    : std::logic_error(build_debug_error_msg(d)), details_(std::move(d))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct a runtime exception from a validation failure.
 */
RuntimeError RuntimeError::from_validate(std::string what,
                                         char const* code,
                                         char const* file,
                                         int line)
{
    return RuntimeError{{RuntimeErrorType::validate, what, code, file, line}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a runtime exception from a CUDA/HIP runtime failure.
 */
RuntimeError RuntimeError::from_device_call(char const* error_string,
                                            char const* code,
                                            char const* file,
                                            int line)
{
    return RuntimeError{
        {RuntimeErrorType::device, error_string, code, file, line}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a message and throw an error from a runtime MPI failure.
 */
RuntimeError RuntimeError::from_mpi_call([[maybe_unused]] int errorcode,
                                         char const* code,
                                         char const* file,
                                         int line)
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
    return RuntimeError{
        {RuntimeErrorType::mpi, error_string, code, file, line}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct an error message from a Geant4 exception.
 *
 * \param origin Usually the function that throws
 * \param code A computery error code
 * \param desc Description of the failure
 */
RuntimeError RuntimeError::from_geant_exception(char const* origin,
                                                char const* code,
                                                char const* desc)
{
    return RuntimeError{{RuntimeErrorType::geant, desc, code, origin, 0}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct an error message from a Geant4 exception.
 *
 * \param origin Usually the function that throws
 * \param msg Description of the failure
 */
RuntimeError RuntimeError::from_root_error(char const* origin, char const* msg)
{
    return RuntimeError{{RuntimeErrorType::root, msg, nullptr, origin, 0}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a runtime error from detailed descriptions.
 */
RuntimeError::RuntimeError(RuntimeErrorDetails d)
    : std::runtime_error(build_runtime_error_msg(d)), details_(std::move(d))
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
