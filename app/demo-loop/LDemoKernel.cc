//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoKernel.cc
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
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Sample mean free path and calculate physics step limits.
 */
void pre_step(const ParamsHostRef& params, const StateHostRef& states)
{
    for (auto tid : range(ThreadId{states.size()}))
    {
        SimTrackView sim(states.sim, tid);
        if (!sim.alive())
            continue;

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
}

//---------------------------------------------------------------------------//
/*!
 * Propagate and process physical changes to the track along the step and
 * select the process/model for discrete interaction.
 */
void along_and_post_step(const ParamsHostRef& params,
                         const StateHostRef&  states)
{
    for (auto tid : range(ThreadId{states.size()}))
    {
        SimTrackView sim(states.sim, tid);
        if (!sim.alive())
        {
            // Clear the model ID so inactive tracks will exit the interaction
            // kernels
            PhysicsTrackView phys(params.physics, states.physics, {}, {}, tid);
            phys.model_id({});
            continue;
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
}

//---------------------------------------------------------------------------//
/*!
 * Postprocessing of secondaries and interaction results.
 */
void process_interactions(const ParamsHostRef& params,
                          const StateHostRef&  states)
{
    for (auto tid : range(ThreadId{states.size()}))
    {
        SimTrackView sim(states.sim, tid);
        if (!sim.alive())
            continue;

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
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
