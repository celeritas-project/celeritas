//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Physics.test.cu
//---------------------------------------------------------------------------//
#include "Physics.test.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void phys_test_kernel(const PTestInput inp)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= inp.states.size())
        return;

    auto const& init = inp.inits[tid];
    PhysicsTrackView phys(inp.params, inp.states, init.particle, init.mat, tid);
    PhysicsStepView step(inp.params, inp.states, tid);

    phys = PhysicsTrackInitializer{};
    inp.result[tid.get()] = calc_step(phys, step, init.energy);
}
}  // namespace

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
void phys_cuda_test(PTestInput const& input)
{
    CELER_ASSERT(input.inits.size() == input.states.size());

    CELER_LAUNCH_KERNEL(
        phys_test, device().default_block_size(), input.states.size(), input);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
