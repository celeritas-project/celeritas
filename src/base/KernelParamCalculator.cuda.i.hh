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
CELER_INLINE_FUNCTION auto KernelParamCalculator::thread_id() -> dim_type
{
#ifdef __CUDA_ARCH__
    return blockIdx.x * blockDim.x + threadIdx.x;
#else
    return 0;
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
