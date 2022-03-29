//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LDemoLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Quantity.hh"
#include "base/StackAllocator.hh"
#include "geometry/GeoMaterialView.hh"
#include "geometry/GeoTrackView.hh"
#include "geometry/LinearPropagator.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsStepUtils.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/material/MaterialTrackView.hh"
#include "random/RngEngine.hh"
#include "random/distributions/ExponentialDistribution.hh"
#include "sim/CoreTrackData.hh"
#include "sim/SimTrackView.hh"

#ifndef CELER_DEVICE_COMPILE
#    include "base/ArrayIO.hh"
#    include "comm/Logger.hh"
#endif

using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::value_as;

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
        using ParamsRef                                                     \
            = celeritas::CoreParamsData<Ownership::const_reference, M>;     \
        using StateRef = celeritas::CoreStateData<Ownership::reference, M>; \
        using ThreadId = celeritas::ThreadId;                               \
                                                                            \
      public:                                                               \
        CELER_FUNCTION                                                      \
        NAME##Launcher(const ParamsRef& params, const StateRef& states)     \
            : params_(params), states_(states)                              \
        {                                                                   \
            CELER_EXPECT(params_);                                          \
            CELER_EXPECT(states_);                                          \
        }                                                                   \
                                                                            \
        inline CELER_FUNCTION void operator()(ThreadId tid) const;          \
                                                                            \
      private:                                                              \
        const ParamsRef& params_;                                           \
        const StateRef&  states_;                                           \
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
    using celeritas::Action;
    using celeritas::ExponentialDistribution;
    using celeritas::real_type;

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

    // Sample mean free path
    if (!phys.has_interaction_mfp())
    {
        celeritas::RngEngine               rng(states_.rng, tid);
        ExponentialDistribution<real_type> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }

    // Calculate physics step limits and total macro xs
    real_type step = calc_tabulated_physics_step(mat, particle, phys);
    if (particle.is_stopped())
    {
        if (phys.macro_xs() == 0)
        {
            // If the particle is stopped and cannot undergo a discrete
            // interaction, kill it
            states_.interactions[tid].action = Action::cutoff_energy;
            sim.alive(false);
            return;
        }
        // Set the interaction length and mfp to zero for active stopped
        // particles
        step = 0;
        phys.interaction_mfp(0);
    }
    phys.step_length(step);
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

    using Energy           = celeritas::ParticleTrackView::Energy;
    using Action           = celeritas::Action;
    using real_type        = celeritas::real_type;
    using LinearPropagator = celeritas::LinearPropagator;
    using ModelId          = celeritas::ModelId;

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
    celeritas::RngEngine         rng(states_.rng, tid);

    bool      crossed_boundary = false;
    real_type step             = phys.step_length();
    real_type start_energy     = value_as<Energy>(particle.energy());

    if (!particle.is_stopped())
    {
        // Boundary crossings and energy loss only need to be considered when
        // the particle is moving
        CELER_ASSERT(step > 0);
        {
            // Propagate up to the step length or next boundary
            LinearPropagator propagate(&geo);
            auto             geo_step = propagate(step);
            step                      = geo_step.distance;
            crossed_boundary          = geo_step.boundary;
        }

        // Calculate energy loss over the step length
        celeritas::CutoffView cutoffs(params_.cutoffs, mat.material_id());
        real_type             eloss = value_as<Energy>(
            calc_energy_loss(cutoffs, mat, particle, phys, step, rng));
        states_.energy_deposition[tid] += eloss;
        particle.energy(Energy{start_energy - eloss});
    }

    ModelId result_model{};
    Action  result_action = Action::unchanged;

    if (particle.is_stopped())
    {
        if (CELER_UNLIKELY(crossed_boundary))
        {
            // Particle should *not* go to zero energy at exactly the same time
            // as it crosses the volume boundary.  Back particle slightly away
            // from boundary to avoid on-surface initialization/direction
            // change.

            real_type backward_bump = real_type(-1e-5) * step;
#ifndef CELER_DEVICE_COMPILE
            using VGT             = celeritas::ValueGridType;
            using RangeCalculator = celeritas::RangeCalculator;

            real_type range = -1;
            if (auto ppid = phys.eloss_ppid())
            {
                auto grid_id = phys.value_grid(VGT::range, ppid);
                auto calc_range
                    = phys.make_calculator<RangeCalculator>(grid_id);
                range = calc_range(Energy{start_energy});
            }

            CELER_LOG(error) << "Track " << sim.track_id().unchecked_get()
                             << " (particle type ID "
                             << particle.particle_id().unchecked_get()
                             << ") lost all energy (" << start_energy
                             << " MeV) while leaving volume "
                             << geo.volume_id().unchecked_get() << " at point "
                             << geo.pos() << " with step length " << step
                             << " cm even though its max range was " << range
                             << " cm.  Bumping by " << backward_bump
                             << " to move it back inside the boundary.";
#endif
            celeritas::Real3 pos = geo.pos();
            axpy(backward_bump, geo.dir(), &pos);
            geo.move_internal(pos);
            crossed_boundary = false;
        }

        if (!phys.has_at_rest())
        {
            // Immediately kill stopped particles with no at rest processes
            result_action = Action::cutoff_energy;
            sim.alive(false);
        }
        else
        {
            // Particle slowed down to zero: force an interaction now
            phys.interaction_mfp(0);
        }
    }
    else
    {
        // Reduce the remaining mean free path
        // TODO: use corresponding step limiter/acceptance to choose action
        real_type mfp = phys.interaction_mfp() - step * phys.macro_xs();
        phys.interaction_mfp(celeritas::soft_zero(mfp) ? 0 : mfp);
    }

    if (crossed_boundary)
    {
        // Particle entered a new volume before reaching the interaction point
        geo.cross_boundary();
        if (geo.is_outside())
        {
            // Kill the track if it's outside the valid geometry region
            result_action = Action::escaped;
            sim.alive(false);
        }
        else
        {
            // Update the material if it's inside
            result_action = Action::entered_volume;
            auto matid    = geo_mat.material_id(geo.volume_id());
            CELER_ASSERT(matid);
            mat = {matid};
        }
    }
    else if (phys.interaction_mfp() <= 0)
    {
        // Reached the interaction point: sample the process and determine
        // the corresponding model
        auto ppid_mid = select_process_and_model(particle, phys, rng);
        result_model  = ppid_mid.model;
    }
    phys.model_id(result_model);
    states_.interactions[tid].action = result_action;
    states_.step_length[tid]         = step;
}

//---------------------------------------------------------------------------//
/*!
 * Postprocess secondaries and interaction results.
 */
template<MemSpace M>
CELER_FUNCTION void
ProcessInteractionsLauncher<M>::operator()(ThreadId tid) const
{
    using Energy = celeritas::ParticleTrackView::Energy;

    celeritas::SimTrackView sim(states_.sim, tid);

    // Increment the step count before checking if the track is alive as some
    // active tracks might have been killed earlier in the step.
    sim.increment_num_steps();

    if (!sim.alive())
    {
        return;
    }

    celeritas::ParticleTrackView particle(
        params_.particles, states_.particles, tid);
    celeritas::GeoTrackView geo(params_.geometry, states_.geometry, tid);

    // Update the track state from the interaction
    // TODO: handle recoverable errors
    const celeritas::Interaction& result = states_.interactions[tid];
    CELER_ASSERT(result);
    if (action_killed(result.action))
    {
        sim.alive(false);
    }
    else if (!action_unchanged(result.action))
    {
        particle.energy(result.energy);
        geo.set_dir(result.direction);
    }

    // Deposit energy from interaction
    states_.energy_deposition[tid]
        += value_as<Energy>(result.energy_deposition);
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void CleanupLauncher<M>::operator()(ThreadId tid) const
{
    CELER_ASSERT(tid.get() == 0);
    celeritas::StackAllocator<celeritas::Secondary> allocate_secondaries(
        states_.secondaries);
    allocate_secondaries.clear();
}

//---------------------------------------------------------------------------//
#undef CDL_LAUNCHER
} // namespace demo_loop
