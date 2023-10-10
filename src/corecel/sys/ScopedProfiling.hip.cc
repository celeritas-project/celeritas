//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.hip.cc
//! \brief The roctx implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//

#include "ScopedProfiling.hh"

#include <roctracer/roctx.h>

#include "corecel/io/Logger.hh"

#include "celeritas_sys_config.h"

#include "Environment.hh"

namespace celeritas
{

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
#ifdef CELERITAS_HAVE_ROCTX
    if (ScopedProfiling::enable_profiling())
    {
        roctxRangePush(input.name.c_str());
    }
#else
    CELER_DISCARD(input);
#endif
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
#ifdef CELERITAS_HAVE_ROCTX
    if (ScopedProfiling::enable_profiling())
    {
        roctxRangePop();
    }
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
