//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.thrust.cu
//---------------------------------------------------------------------------//
#include "KNDemoKernel.hh"

#include "base/device_runtime_api.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace celeritas;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Sum the total number of living particles.
 */
size_type reduce_alive(const DeviceGridParams& grid, Span<const bool> alive)
{
    size_type result = thrust::reduce(
        thrust::device_pointer_cast(alive.data()),
        thrust::device_pointer_cast(alive.data() + alive.size()),
        size_type(0),
        thrust::plus<size_type>());
    CELER_DEVICE_CHECK_ERROR();

    if (grid.sync)
    {
        CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
