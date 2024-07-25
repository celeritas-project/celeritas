//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.hip.cc
//! \brief The roctx implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

#include <string>

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"

#include "Environment.hh"

#if CELERITAS_HAVE_ROCTX
#    include <roctracer/roctx.h>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Activate profiling.
 */
void ScopedProfiling::activate(Input const& input) noexcept
{
    int result = 0;
#if CELERITAS_HAVE_ROCTX
    std::string temp_name{input.name};
    result = roctxRangePush(temp_name.c_str());
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
