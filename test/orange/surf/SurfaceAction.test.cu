//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceAction.test.cu
//---------------------------------------------------------------------------//
#include "SurfaceAction.test.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void sa_test_kernel(SATestInput input)
{
    auto tid = KernelParamCalculator::thread_id();
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
    CELER_LAUNCH_KERNEL(
        sa_test, device().default_block_size(), input.states.size(), input);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
