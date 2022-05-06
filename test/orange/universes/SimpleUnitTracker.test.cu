//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/universes/SimpleUnitTracker.test.cu
//---------------------------------------------------------------------------//
#include "SimpleUnitTracker.test.hh"

#include "corecel/sys/KernelParamCalculator.device.hh"

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
    CELER_LAUNCH_KERNEL(initialize,
                        celeritas::device().default_block_size(),
                        state.size(),
                        params,
                        state);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
