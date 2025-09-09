//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/SimpleUnitTracker.test.cu
//---------------------------------------------------------------------------//
#include "SimpleUnitTracker.test.hh"

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

__global__ void initialize_kernel(ParamsRef<MemSpace::device> const params,
                                  StateRef<MemSpace::device> const states)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    InitializingExecutor<> calc_thread{params, states};
    calc_thread(TrackSlotId{tid.unchecked_get()});
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void test_initialize(ParamsRef<MemSpace::device> const& params,
                     StateRef<MemSpace::device> const& state)
{
    CELER_LAUNCH_KERNEL(initialize, state.size(), 0, params, state);
    CELER_DEVICE_API_CALL(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
