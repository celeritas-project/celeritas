//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SurfaceAction.test.cu
//---------------------------------------------------------------------------//
#include "SurfaceAction.test.hh"

#include "base/KernelParamCalculator.cuda.hh"

namespace celeritas_test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void sa_test_kernel(SATestInput input)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= input.states.size())
        return;

    // Calculate distances in parallel
    CalcSenseDistanceLauncher<> calc_thread{input.params, input.states};
    calc_thread(tid);
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void sa_test(SATestInput input)
{
    static const celeritas::KernelParamCalculator calc_launch_params(
        sa_test_kernel, "sa_test");
    auto params = calc_launch_params(input.states.size());
    sa_test_kernel<<<params.grid_size, params.block_size>>>(input);

    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
