//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimpleUnitTracker.test.cu
//---------------------------------------------------------------------------//
#include "SimpleUnitTracker.test.hh"

#include <thrust/reduce.h>
#include "base/KernelParamCalculator.cuda.hh"

namespace celeritas_test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void initialize_kernel(const ParamsRef<MemSpace::device> params,
                                  const StateRef<MemSpace::device>  states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    InitializingLauncher<> calc_thread{params, states};
    calc_thread(tid);
}
} // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void test_initialize(const ParamsRef<MemSpace::device>& params,
                     const StateRef<MemSpace::device>&  state)
{
    static const celeritas::KernelParamCalculator calc_launch_params(
        initialize_kernel, "initialize");
    auto launch = calc_launch_params(state.size());
    initialize_kernel<<<launch.grid_size, launch.block_size>>>(params, state);

    CELER_CUDA_CHECK_ERROR();
    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
