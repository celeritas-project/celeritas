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
            if (CELERITAS_HAVE_ROCTX)
            {
                CELER_LOG(info)
                    << "Enabling profiling support since the "
                       "'CELER_ENABLE_PROFILING' "
                       "environment variable is present and non-empty";
            }
            else
            {
                CELER_LOG(warning)
                    << "Roctx library not found. ScopedProfiling "
                       "has no effect";
            }
            return true;
        }
        return false;
    }();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Activate profiling.
 */
ScopedProfiling::activate_(Input const& input)
{
#if CELERITAS_HAVE_ROCTX
    roctxRangePush(input.name.c_str());
#else
    CELER_DISCARD(input);
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
/*!
 * End the profiling range.
 */
ScopedProfiling::deactivate_()
{
#if CELERITAS_HAVE_ROCTX
    roctxRangePop();
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
