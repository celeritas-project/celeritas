//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimpleUnitTracker.test.cu
//---------------------------------------------------------------------------//
#include "SimpleUnitTracker.test.hh"

#include "base/KernelParamCalculator.cuda.hh"

namespace celeritas_test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void
initialize_kernel(const ParamsDeviceRef params, const StateDeviceRef states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= input.states.size())
        return;

    InitializingLauncher<> calc_thread{input.params, input.states};
    calc_thread(tid);
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void test_initialize(ParamsDeviceRef, StateDeviceRef)
{
    static const celeritas::KernelParamCalculator calc_launch_params(
        initialize_kernel, "initialize");
    auto params = calc_launch_params(input.num_threads);
    initialize_kernel<<<params.grid_size, params.block_size>>>(
        input.num_threads);

    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
