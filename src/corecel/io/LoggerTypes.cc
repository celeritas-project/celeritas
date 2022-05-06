//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/LoggerTypes.cc
//---------------------------------------------------------------------------//
#include "LoggerTypes.hh"

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the plain text equivalent of the LogLevel enum.
 */
const char* to_cstring(LogLevel lev)
{
    static const char* const levels[] = {
        "debug",
        "diagnostic",
        "status",
        "info",
        "warning",
        "error",
        "critical",
    };
    auto idx = static_cast<unsigned int>(lev);
    CELER_ENSURE(idx * sizeof(const char*) < sizeof(levels));
    return levels[idx];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
