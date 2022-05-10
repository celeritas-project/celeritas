//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-loop/diagnostic/ParticleProcessDiagnostic.cu
//---------------------------------------------------------------------------//
#include "ParticleProcessDiagnostic.hh"

#include "corecel/sys/KernelParamCalculator.device.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Tally the particle/process combinations that occur at each step.
 */
__global__ void count_particle_process_kernel(
    CoreParamsDeviceRef const                           params,
    CoreStateDeviceRef const                            states,
    ParticleProcessLauncher<MemSpace::device>::ItemsRef counts)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    ParticleProcessLauncher<MemSpace::device> launch(params, states, counts);
    launch(tid);
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Launch kernel to tally the particle/process combinations.
 */
void count_particle_process(
    const CoreParamsDeviceRef&                          params,
    const CoreStateDeviceRef&                           states,
    ParticleProcessLauncher<MemSpace::device>::ItemsRef counts)
{
    static const KernelParamCalculator calc_launch_params(
        count_particle_process_kernel, "count_particle_process");
    auto kp = calc_launch_params(states.size());
    count_particle_process_kernel<<<kp.blocks_per_grid, kp.threads_per_block>>>(
        params, states, counts);
    CELER_DEVICE_CHECK_ERROR();
}
//---------------------------------------------------------------------------//
} // namespace demo_loop
