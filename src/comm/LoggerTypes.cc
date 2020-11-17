//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LoggerTypes.cc
//---------------------------------------------------------------------------//
#include "LoggerTypes.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
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
    int idx = static_cast<int>(lev);
    ENSURE(idx * sizeof(const char*) < sizeof(levels));
    return levels[idx];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
