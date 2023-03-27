//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LoggerTypes.cc
//---------------------------------------------------------------------------//
#include "LoggerTypes.hh"

#include "corecel/io/EnumStringMapper.hh"

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
}  // namespace celeritas
