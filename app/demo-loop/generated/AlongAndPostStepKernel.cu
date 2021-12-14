//----------------------------------*-cu-*-----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AlongAndPostStepKernel.cu
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
__global__ void along_and_post_step_kernel(
    ParamsDeviceRef const params,
    StateDeviceRef const states)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    AlongAndPostStepLauncher<MemSpace::device> launch(params, states);
    launch(tid);
}
} // namespace

void along_and_post_step(
    const celeritas::ParamsDeviceRef& params,
    const celeritas::StateDeviceRef& states)
{
    CELER_EXPECT(params);
    CELER_EXPECT(states);

    static const KernelParamCalculator along_and_post_step_ckp(
        along_and_post_step_kernel, "along_and_post_step");
    auto kp = along_and_post_step_ckp(states.size());
    along_and_post_step_kernel<<<kp.grid_size, kp.block_size>>>(
        params, states);
    CELER_CUDA_CHECK_ERROR();
}

} // namespace generated
} // namespace demo_loop
