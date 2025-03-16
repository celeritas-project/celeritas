//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/LocalSurfaceVisitor.test.cu
//---------------------------------------------------------------------------//
#include "LocalSurfaceVisitor.test.hh"

#include "corecel/DeviceRuntimeApi.hh"

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
    CalcSenseDistanceExecutor<> calc_thread{input.params, input.states};
    calc_thread(TrackSlotId{tid.unchecked_get()});
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void sa_test(SATestInput input)
{
    CELER_LAUNCH_KERNEL(sa_test, input.states.size(), 0, input);
    CELER_DEVICE_API_CALL(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
