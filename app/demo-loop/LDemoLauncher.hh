//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/StackAllocator.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/base/CutoffView.hh"
#include "random/RngEngine.hh"
#include "sim/SimTrackView.hh"
#include "sim/TrackData.hh"
#include "KernelUtils.hh"

using celeritas::MemSpace;
using celeritas::Ownership;

namespace demo_loop
{
//---------------------------------------------------------------------------//
// LAUNCHERS
//---------------------------------------------------------------------------//
#define CDL_LAUNCHER(NAME)                                                  \
    template<MemSpace M>                                                    \
    class NAME##Launcher                                                    \
    {                                                                       \
      public:                                                               \
        using ParamsDataRef                                                 \
            = celeritas::ParamsData<Ownership::const_reference, M>;         \
        using StateDataRef = celeritas::StateData<Ownership::reference, M>; \
        using ThreadId     = celeritas::ThreadId;                           \
                                                                            \
      public:                                                               \
        CELER_FUNCTION NAME##Launcher(const ParamsDataRef& params,          \
                                      const StateDataRef&  states)          \
            : params_(params), states_(states)                              \
        {                                                                   \
            CELER_EXPECT(params_);                                          \
            CELER_EXPECT(states_);                                          \
        }                                                                   \
                                                                            \
        inline CELER_FUNCTION void operator()(ThreadId tid) const;          \
                                                                            \
      private:                                                              \
        const ParamsDataRef& params_;                                       \
        const StateDataRef&  states_;                                       \
    };

CDL_LAUNCHER(PreStep)
CDL_LAUNCHER(AlongAndPostStep)
CDL_LAUNCHER(ProcessInteractions)
CDL_LAUNCHER(Cleanup)

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Pre-step logic.
 *
 * Sample the mean free path and calculate the physics step limits.
 */
template<MemSpace M>
CELER_FUNCTION void PreStepLauncher<M>::operator()(ThreadId tid) const
{
    // Clear out energy deposition
    states_.energy_deposition[tid] = 0;

    celeritas::SimTrackView sim(states_.sim, tid);
    if (!sim.alive())
        return;

    celeritas::ParticleTrackView particle(
        params_.particles, states_.particles, tid);
    celeritas::GeoTrackView      geo(params_.geometry, states_.geometry, tid);
    celeritas::GeoMaterialView   geo_mat(params_.geo_mats);
    celeritas::MaterialTrackView mat(params_.materials, states_.materials, tid);
    celeritas::PhysicsTrackView  phys(params_.physics,
                                     states_.physics,
                                     particle.particle_id(),
                                     geo_mat.material_id(geo.volume_id()),
                                     tid);
    celeritas::RngEngine         rng(states_.rng, ThreadId(tid));

    // Sample mfp and calculate minimum step (interaction or step-limited)
    demo_loop::calc_step_limits(
        mat, particle, phys, sim, rng, &states_.interactions[tid]);
}

//---------------------------------------------------------------------------//
/*!
 * Combined along- and post-step logic.
 *
 * Propagate and process physical changes to the track along the step and
 * select the process/model for discrete interaction.
 */
template<MemSpace M>
CELER_FUNCTION void AlongAndPostStepLauncher<M>::operator()(ThreadId tid) const
{
    celeritas::SimTrackView sim(states_.sim, tid);
    if (!sim.alive())
    {
        // Clear the model ID so inactive tracks will exit the interaction
        // kernels
        celeritas::PhysicsTrackView phys(
            params_.physics, states_.physics, {}, {}, tid);
        phys.model_id({});
        return;
    }

    celeritas::ParticleTrackView particle(
        params_.particles, states_.particles, tid);
    celeritas::GeoTrackView      geo(params_.geometry, states_.geometry, tid);
    celeritas::GeoMaterialView   geo_mat(params_.geo_mats);
    celeritas::MaterialTrackView mat(params_.materials, states_.materials, tid);
    celeritas::PhysicsTrackView  phys(params_.physics,
                                     states_.physics,
                                     particle.particle_id(),
                                     geo_mat.material_id(geo.volume_id()),
                                     tid);
    celeritas::CutoffView        cutoffs(params_.cutoffs, mat.material_id());
    celeritas::RngEngine         rng(states_.rng, ThreadId(tid));

    // Propagate, calculate energy loss, and select model
    demo_loop::move_and_select_model(cutoffs,
                                     geo_mat,
                                     geo,
                                     mat,
                                     particle,
                                     phys,
                                     sim,
                                     rng,
                                     &states_.energy_deposition[tid],
                                     &states_.interactions[tid]);
}

//---------------------------------------------------------------------------//
/*!
 * Postprocess secondaries and interaction results.
 */
template<MemSpace M>
CELER_FUNCTION void
ProcessInteractionsLauncher<M>::operator()(ThreadId tid) const
{
    celeritas::SimTrackView sim(states_.sim, tid);
    if (!sim.alive())
        return;

    celeritas::ParticleTrackView particle(
        params_.particles, states_.particles, tid);
    celeritas::GeoTrackView     geo(params_.geometry, states_.geometry, tid);
    celeritas::GeoMaterialView  geo_mat(params_.geo_mats);
    celeritas::PhysicsTrackView phys(params_.physics,
                                     states_.physics,
                                     particle.particle_id(),
                                     geo_mat.material_id(geo.volume_id()),
                                     tid);

    // Apply interaction change
    demo_loop::post_process(geo,
                            particle,
                            phys,
                            sim,
                            &states_.energy_deposition[tid],
                            states_.interactions[tid]);
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void CleanupLauncher<M>::operator()(ThreadId tid) const
{
    CELER_ASSERT(tid.get() == 0);
    StackAllocator<Secondary> allocate_secondaries(states_.secondaries);
    allocate_secondaries.clear();
}
//---------------------------------------------------------------------------//
#undef CDL_LAUNCHER
} // namespace demo_loop
