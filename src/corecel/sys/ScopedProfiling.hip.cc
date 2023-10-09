//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.hip.cc
//---------------------------------------------------------------------------//

#include "ScopedProfiling.hh"

#include <roctracer/roctx.h>

#include "corecel/io/Logger.hh"

#include "Environment.hh"

/**
 * @file
 *
 * The roctx implementation of \c ScopedProfiling
 */

namespace celeritas
{
namespace
{

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Whether profiling is enabled.
 *
 * This is true only if the \c CELER_ENABLE_PROFILING environment variable is
 * set to a non-empty value.
 */
bool ScopedProfiling::enable_profiling()
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_ENABLE_PROFILING").empty())
        {
            CELER_LOG(info) << "Enabling profiling support since the "
                               "'CELER_ENABLE_PROFILING' "
                               "environment variable is present and non-empty";
            return true;
        }
        return false;
    }();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Activate nvtx profiling with options.
 */
ScopedProfiling::ScopedProfiling(Input input)
{
    if (ScopedProfiling::enable_profiling())
    {
        roctxRangePush(input.name.c_str());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Activate nvtx profiling.
 */
ScopedProfiling::ScopedProfiling(std::string const& name)
    : ScopedProfiling{Input{name}}
{
}

//---------------------------------------------------------------------------//
/*!
 * End the profiling range.
 */
ScopedProfiling::~ScopedProfiling()
{
    if (ScopedProfiling::enable_profiling())
    {
        roctxRangePop();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
