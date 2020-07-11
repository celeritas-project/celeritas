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
CELER_FUNCTION auto KernelParamCalculator::thread_id() -> ThreadId
{
#ifdef __CUDA_ARCH__
    return ThreadId{blockIdx.x * blockDim.x + threadIdx.x};
#else
    return ThreadId{0u};
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
