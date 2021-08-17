//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.cu
//---------------------------------------------------------------------------//
#include "LDemoKernel.hh"

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
    GeoMaterialView   geo_mat(params.geo_mats);
    MaterialTrackView mat(params.materials, states.materials, tid);
    PhysicsTrackView  phys(params.physics,
                          states.physics,
                          particle.particle_id(),
                          geo_mat.material_id(geo.volume_id()),
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
    {
        // Clear the model ID so inactive tracks will exit the interaction
        // kernels
        PhysicsTrackView phys(params.physics, states.physics, {}, {}, tid);
        phys.model_id({});
        return;
    }

    ParticleTrackView particle(params.particles, states.particles, tid);
    GeoTrackView      geo(params.geometry, states.geometry, tid);
    GeoMaterialView   geo_mat(params.geo_mats);
    MaterialTrackView mat(params.materials, states.materials, tid);
    PhysicsTrackView  phys(params.physics,
                          states.physics,
                          particle.particle_id(),
                          geo_mat.material_id(geo.volume_id()),
                          tid);
    CutoffView        cutoffs(params.cutoffs, mat.material_id());
    RngEngine         rng(states.rng, ThreadId(tid));

    // Propagate, calculate energy loss, and select model
    demo_loop::move_and_select_model(cutoffs,
                                     geo_mat,
                                     geo,
                                     mat,
                                     particle,
                                     phys,
                                     sim,
                                     rng,
                                     &states.energy_deposition[tid],
                                     &states.interactions[tid]);
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
    GeoMaterialView   geo_mat(params.geo_mats);
    PhysicsTrackView  phys(params.physics,
                          states.physics,
                          particle.particle_id(),
                          geo_mat.material_id(geo.volume_id()),
                          tid);
    CutoffView        cutoffs(params.cutoffs, mat.material_id());

    // Apply cutoffs and interaction change
    demo_loop::post_process(cutoffs,
                            geo,
                            particle,
                            phys,
                            sim,
                            &states.energy_deposition[tid],
                            states.interactions[tid]);
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
 * Clear secondaries.
 */
void cleanup(const ParamsDeviceRef& params, const StateDeviceRef& states)
{
    CDL_LAUNCH_KERNEL(cleanup, 1, params, states);
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
