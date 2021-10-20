//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleProcessDiagnostic.cu
//---------------------------------------------------------------------------//
#include "ParticleProcessDiagnostic.hh"

#include "base/KernelParamCalculator.cuda.hh"

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
    ParamsDeviceRef const                                         params,
    StateDeviceRef const                                          states,
    Collection<size_type, Ownership::reference, MemSpace::device> counts)
{
    auto tid = KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    auto model_id = states.physics.state[tid].model_id;
    if (model_id)
    {
        size_type index = model_id.get() * params.physics.process_groups.size()
                          + states.particles.state[tid].particle_id.get();
        CELER_ASSERT(index < counts.size());
        atomic_add(&counts[ItemId<size_type>(index)], 1u);
    }
}

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Launch kernel to tally the particle/process combinations.
 */
void count_particle_process(
    const ParamsDeviceRef&                                        params,
    const StateDeviceRef&                                         states,
    Collection<size_type, Ownership::reference, MemSpace::device> counts)
{
    static const KernelParamCalculator calc_launch_params(
        count_particle_process_kernel, "count_particle_process");
    auto kp = calc_launch_params(states.size());
    count_particle_process_kernel<<<kp.grid_size, kp.block_size>>>(
        params, states, counts);
    CELER_CUDA_CHECK_ERROR();
}
//---------------------------------------------------------------------------//
} // namespace demo_loop
