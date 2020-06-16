//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelParamCalculator.i.cuh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the linear thread ID.
 */
__device__ KernelParamCalculator::dim_type KernelParamCalculator::thread_id()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
