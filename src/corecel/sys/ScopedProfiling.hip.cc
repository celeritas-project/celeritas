//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.hip.cc
//! \brief The roctx implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

#include "celeritas_sys_config.h"
#include "corecel/io/Logger.hh"

#include "Environment.hh"

#if CELERITAS_HAVE_ROCTX
#    include <roctracer/roctx.h>
#endif

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Whether profiling is enabled.
 *
 * This is true only if the \c CELER_ENABLE_PROFILING environment variable is
 * set to a non-empty value.
 */
bool ScopedProfiling::use_profiling()
{
    static bool const result = [] {
        if (!celeritas::getenv("CELER_ENABLE_PROFILING").empty())
        {
            if (!CELERITAS_HAVE_ROCTX)
            {
                CELER_LOG(warning) << "Disabling profiling support "
                                      "since ROC-TX is unavailable";
                return false;
            }
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
 * Activate profiling.
 */
void ScopedProfiling::activate(Input const& input) noexcept
{
    int result = 0;
#if CELERITAS_HAVE_ROCTX
    result = roctxRangePush(input.name.c_str());
#endif
    if (result < 0)
    {
        activated_ = false;
        CELER_LOG(warning) << "Failed to activate profiling range '"
                           << input.name << "'";
    }
}

//---------------------------------------------------------------------------//
/*!
 * End the profiling range.
 */
void ScopedProfiling::deactivate() noexcept
{
    int result = 0;
#if CELERITAS_HAVE_ROCTX
    result = roctxRangePop();
#endif
    if (result < 0)
    {
        activated_ = false;
        CELER_LOG(warning) << "Failed to deactivate profiling range";
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
