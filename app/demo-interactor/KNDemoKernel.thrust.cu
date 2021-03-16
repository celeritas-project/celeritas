//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.thrust.cu
//---------------------------------------------------------------------------//
#include "KNDemoKernel.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace celeritas;

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Sum the total number of living particles.
 */
size_type reduce_alive(const CudaGridParams& grid, Span<const bool> alive)
{
    size_type result = thrust::reduce(
        thrust::device_pointer_cast(alive.data()),
        thrust::device_pointer_cast(alive.data() + alive.size()),
        size_type(0),
        thrust::plus<size_type>());
    CELER_CUDA_CHECK_ERROR();

    if (grid.sync)
    {
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
