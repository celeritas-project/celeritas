//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
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
    CoreParamsDeviceRef const params,
    CoreStateDeviceRef const states,
    ParticleProcessLauncher<MemSpace::device>::ItemsRef counts)
{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < states.size()))
        return;

    ParticleProcessLauncher<MemSpace::device> launch(params, states, counts);
    launch(TrackSlotId{tid.unchecked_get()});
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Launch kernel to tally the particle/process combinations.
 */
void count_particle_process(
    CoreParamsDeviceRef const& params,
    CoreStateDeviceRef const& states,
    ParticleProcessLauncher<MemSpace::device>::ItemsRef counts)
{
    CELER_LAUNCH_KERNEL(count_particle_process,
                        celeritas::device().default_block_size(),
                        states.size(),
                        params,
                        states,
                        counts);
}
//---------------------------------------------------------------------------//
}  // namespace demo_loop
