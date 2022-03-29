//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StepDiagnostic.cu
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

#include "base/KernelParamCalculator.device.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Count the steps per track for each particle type.
 */
__global__ void count_steps_kernel(CoreParamsDeviceRef const params,
                                   CoreStateDeviceRef const  states,
                                   StepDiagnosticDataRef<MemSpace::device> data)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    StepLauncher<MemSpace::device> launch(params, states, data);
    launch(tid);
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Launch kernel to tally the steps per track.
 */
void count_steps(const CoreParamsDeviceRef&              params,
                 const CoreStateDeviceRef&               states,
                 StepDiagnosticDataRef<MemSpace::device> data)
{
    static const KernelParamCalculator calc_launch_params(count_steps_kernel,
                                                          "count_steps");
    auto                               kp = calc_launch_params(states.size());
    count_steps_kernel<<<kp.blocks_per_grid, kp.threads_per_block>>>(
        params, states, data);
    CELER_DEVICE_CHECK_ERROR();
}
//---------------------------------------------------------------------------//
} // namespace demo_loop
