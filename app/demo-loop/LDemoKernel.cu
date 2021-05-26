//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.cu
//---------------------------------------------------------------------------//
#include "LDemoKernel.hh"

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include "base/KernelParamCalculator.cuda.hh"
#include "base/StackAllocator.hh"
#include "physics/base/CutoffView.hh"
#include "random/RngEngine.hh"
#include "sim/SimTrackView.hh"
#include "KernelUtils.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Whether the track is alive.
 */
struct alive
{
    __device__ size_type operator()(const SimTrackState& sim) const
    {
        return sim.alive ? 1 : 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Sample mean free path and calculate physics step limits.
 */
__global__ void
pre_step_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    SimTrackView sim(states.sim, tid);
    if (!sim.alive())
        return;

    ParticleTrackView particle(params.particles, states.particles, tid);
    GeoTrackView      geo(params.geometry, states.geometry, tid);
    GeoMaterialView   geo_mat(params.geo_mats, geo.volume_id());
    MaterialTrackView mat(params.materials, states.materials, tid);
    PhysicsTrackView  phys(params.physics,
                          states.physics,
                          particle.particle_id(),
                          geo_mat.material_id(),
                          tid);
    RngEngine         rng(states.rng, ThreadId(tid));

    // Sample mfp and calculate minimum step (interaction or step-limited)
    demo_loop::calc_step_limits(mat, particle, phys, sim, rng);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate and process physical changes to the track along the step and
 * select the process/model for discrete interaction.
 */
__global__ void along_and_post_step_kernel(ParamsDeviceRef const params,
                                           StateDeviceRef const  states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    SimTrackView sim(states.sim, tid);
    if (!sim.alive())
        return;

    ParticleTrackView particle(params.particles, states.particles, tid);
    GeoTrackView      geo(params.geometry, states.geometry, tid);
    GeoMaterialView   geo_mat(params.geo_mats, geo.volume_id());
    PhysicsTrackView  phys(params.physics,
                          states.physics,
                          particle.particle_id(),
                          geo_mat.material_id(),
                          tid);
    RngEngine         rng(states.rng, ThreadId(tid));

    // Move particle and determine the actual distance traveled
    real_type step = demo_loop::propagate(geo, phys);

    // Calculate energy loss over the step length
    auto eloss = calc_energy_loss(particle, phys, step);
    states.energy_deposition[tid] += eloss.value();

    // The particle entered a new volume before reaching the interaction
    if (step < phys.step_length())
    {
        states.interactions[tid]
            = Interaction::from_boundary(particle.energy(), geo.dir());
    }

    // TODO: is this right??
    // Kill the track if it's outside the valid geometry region
    if (geo.is_outside())
        sim.alive(false);

    // Select the model for the discrete process
    demo_loop::select_discrete_model(particle, phys, rng, step, eloss);
}

//---------------------------------------------------------------------------//
/*!
 * Postprocessing of secondaries and interaction results.
 */
__global__ void process_interactions_kernel(ParamsDeviceRef const params,
                                            StateDeviceRef const  states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() >= states.size())
        return;

    SimTrackView sim(states.sim, tid);
    if (!sim.alive())
        return;

    ParticleTrackView particle(params.particles, states.particles, tid);
    GeoTrackView      geo(params.geometry, states.geometry, tid);
    MaterialTrackView mat(params.materials, states.materials, tid);
    GeoMaterialView   geo_mat(params.geo_mats, geo.volume_id());
    PhysicsTrackView  phys(params.physics,
                          states.physics,
                          particle.particle_id(),
                          geo_mat.material_id(),
                          tid);
    CutoffView        cutoffs(params.cutoffs, mat.material_id());

    // Update the track state from the interaction
    const Interaction& result = states.interactions[tid];
    if (action_killed(result.action))
    {
        sim.alive(false);
    }
    else if (!action_unchanged(result.action)
             && !action_crossed_boundary(result.action))
    {
        particle.energy(result.energy);
        geo.set_dir(result.direction);
    }

    // Deposit energy from interaction
    states.energy_deposition[tid] += result.energy_deposition.value();

    // Kill secondaries with energy below the production threshold and deposit
    // their energy
    for (auto& secondary : result.secondaries)
    {
        if (secondary.energy < cutoffs.energy(secondary.particle_id))
        {
            states.energy_deposition[tid] += secondary.energy.value();
            secondary = {};
        }
    }

    // Reset the physics state if a discrete interaction occured
    if (phys.model_id())
        phys = {};
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
__global__ void
cleanup_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);

    if (tid.get() == 0)
    {
        allocate_secondaries.clear();
    }
}

} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
#define CDL_LAUNCH_KERNEL(NAME, THREADS, ARGS...)                   \
    do                                                              \
    {                                                               \
        static const ::celeritas::KernelParamCalculator NAME##_ckp( \
            NAME##_kernel, #NAME);                                  \
        auto kp = NAME##_ckp(THREADS);                              \
                                                                    \
        NAME##_kernel<<<kp.grid_size, kp.block_size>>>(ARGS);       \
        CELER_CUDA_CHECK_ERROR();                                   \
    } while (0)

//---------------------------------------------------------------------------//
/*!
 * Get minimum step length from interactions.
 */
void pre_step(const ParamsDeviceRef& params, const StateDeviceRef& states)
{
    CDL_LAUNCH_KERNEL(pre_step, states.size(), params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Propogation, slowing down, and discrete model selection.
 */
void along_and_post_step(const ParamsDeviceRef& params,
                         const StateDeviceRef&  states)
{
    CDL_LAUNCH_KERNEL(along_and_post_step, states.size(), params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Postprocessing of secondaries and interaction results.
 */
void process_interactions(const ParamsDeviceRef& params,
                          const StateDeviceRef&  states)
{
    CDL_LAUNCH_KERNEL(process_interactions, states.size(), params, states);
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of active tracks.
 */
size_type reduce_alive(const StateDeviceRef& states)
{
    auto sim_states = states.sim.state[AllItems<SimTrackState>{}].data();
    return thrust::transform_reduce(
        thrust::device_pointer_cast(sim_states),
        thrust::device_pointer_cast(sim_states) + states.size(),
        alive(),
        0,
        thrust::plus<size_type>());
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
void cleanup(const ParamsDeviceRef& params, const StateDeviceRef& states)
{
    CDL_LAUNCH_KERNEL(cleanup, 1, params, states);
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
