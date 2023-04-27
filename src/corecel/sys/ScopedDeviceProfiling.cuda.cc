//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedDeviceProfiling.cuda.cc
//---------------------------------------------------------------------------//
#include "ScopedDeviceProfiling.hh"

#include "celeritas_config.h"
#include "corecel/device_runtime_api.h"

// Profiler API isn't included with regular CUDA API headers
#include <cuda_profiler_api.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

#include "Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Activate device profiling.
 */
ScopedDeviceProfiling::ScopedDeviceProfiling()
{
    if (celeritas::device())
    {
        try
        {
            CELER_CUDA_CALL(cudaProfilerStart());
        }
        catch (RuntimeError const& e)
        {
            CELER_LOG(error) << "Failed to start profiling: " << e.what();
            return;
        }
        activated_ = true;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Deactivate device profiling if this function activated it.
 */
ScopedDeviceProfiling::~ScopedDeviceProfiling()
{
    if (activated_)
    {
        try
        {
            CELER_CUDA_CALL(cudaProfilerStop());
        }
        catch (RuntimeError const& e)
        {
            CELER_LOG(error) << "Failed to stop profiling: " << e.what();
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
