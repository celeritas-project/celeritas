//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KNDemoKernel.thrust.cu
//---------------------------------------------------------------------------//
#include "KNDemoKernel.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "corecel/device_runtime_api.h"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Sum the total number of living particles.
 */
size_type reduce_alive(DeviceGridParams const& grid, Span<bool const> alive)
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
}  // namespace app
}  // namespace celeritas
