//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Assert.cc
//---------------------------------------------------------------------------//
#include "Assert.hh"

#include <sstream>
#include "ColorUtils.hh"

namespace celeritas
{
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
[[noreturn]] void
throw_debug_error(const char* condition, const char* file, int line)
{
    std::ostringstream msg;
    // clang-format off
    msg << color_code('W') << file << ':' << line << ':'
        << color_code(' ') << "\nceleritas: "
        << color_code('R') << "assertion: "
        << color_code('x') << condition
        << color_code(' ');
    // clang-format on
    throw DebugError(msg.str());
}

//---------------------------------------------------------------------------//
/*!
 * Construct a debug assertion message and throw.
 */
[[noreturn]] void throw_cuda_call_error(const char* error_string,
                                        const char* code,
                                        const char* file,
                                        int         line)
{
    std::ostringstream msg;
    // clang-format off
    msg << color_code('W') << file << ':' << line << ':'
        << color_code(' ') << "\nceleritas: "
        << color_code('R') << "cuda error: "
        << color_code(' ') << error_string << "\n    "
        << color_code('x') << code
        << color_code(' ');
    // clang-format on
    throw RuntimeError(msg.str());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
