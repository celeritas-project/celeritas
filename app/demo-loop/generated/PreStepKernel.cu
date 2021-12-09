//----------------------------------*-cu-*-----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PreStepKernel.cu
//! \note Auto-generated by gen-demo-loop-kernel.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "base/Assert.hh"
#include "base/Types.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "../LDemoLauncher.hh"

using namespace celeritas;

namespace demo_loop
{
namespace generated
{
namespace
{
__global__ void
pre_step_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    PreStepLauncher<MemSpace::device> launch(params, states);
    launch(tid);
}
} // namespace

void pre_step(const celeritas::ParamsDeviceRef& params,
              const celeritas::StateDeviceRef&  states)
{
    CELER_EXPECT(params);
    CELER_EXPECT(states);

    static const KernelParamCalculator pre_step_ckp(pre_step_kernel,
                                                    "pre_step");
    auto                               kp = pre_step_ckp(states.size());
    pre_step_kernel<<<kp.grid_size, kp.block_size>>>(params, states);
    CELER_CUDA_CHECK_ERROR();
}

} // namespace generated
} // namespace demo_loop
