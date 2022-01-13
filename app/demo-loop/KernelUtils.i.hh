//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelUtils.i.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "geometry/LinearPropagator.hh"
#include "random/distributions/ExponentialDistribution.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Sample mean free path and calculate physics step limits.
 */
template<class Rng>
CELER_FUNCTION void calc_step_limits(const MaterialTrackView& mat,
                                     const ParticleTrackView& particle,
                                     PhysicsTrackView&        phys,
                                     SimTrackView&            sim,
                                     Rng&                     rng,
                                     Interaction*             result)
{
    // Sample mean free path
    if (!phys.has_interaction_mfp())
    {
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
            result->action = Action::cutoff_energy;
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
 * Propagate up to the step length or next boundary, calculate the energy loss
 * over the step, and select the model for the discrete interaction.
 */
template<class Rng>
CELER_FUNCTION void move_and_select_model(const CutoffView&      cutoffs,
                                          const GeoMaterialView& geo_mat,
                                          GeoTrackView&          geo,
                                          MaterialTrackView&     mat,
                                          ParticleTrackView&     particle,
                                          PhysicsTrackView&      phys,
                                          SimTrackView&          sim,
                                          Rng&                   rng,
                                          real_type*             edep,
                                          Interaction*           result)
{
    using Energy = ParticleTrackView::Energy;

    // Actual distance, limited by along-step length or geometry
    real_type step = phys.step_length();
    if (step > 0)
    {
        // Propagate up to the step length or next boundary
        LinearPropagator propagate(&geo);
        auto             geo_step = propagate(step);
        step                      = geo_step.distance;

        // Particle entered a new volume before reaching the interaction point
        if (geo_step.boundary)
        {
            if (geo.is_outside())
            {
                // Kill the track if it's outside the valid geometry region
                result->action = Action::escaped;
                sim.alive(false);
            }
            else
            {
                // Update the material if it's inside
                result->action = Action::entered_volume;
                auto matid     = geo_mat.material_id(geo.volume_id());
                CELER_ASSERT(matid);
                mat = {matid};
            }
        }
    }
    phys.step_length(phys.step_length() - step);

    // Calculate energy loss over the step length
    auto eloss = calc_energy_loss(cutoffs, mat, particle, phys, step, rng);
    *edep += value_as<Energy>(eloss);
    particle.energy(
        Energy{value_as<Energy>(particle.energy()) - value_as<Energy>(eloss)});

    // Reduce the remaining mean free path
    real_type mfp = phys.interaction_mfp() - step * phys.macro_xs();
    phys.interaction_mfp(soft_zero(mfp) ? 0 : mfp);

    ModelId model{};
    if (phys.interaction_mfp() <= 0)
    {
        // Reached the interaction point: sample the process and determine the
        // corresponding model
        auto ppid_mid = select_process_and_model(particle, phys, rng);
        model         = ppid_mid.model;
    }
    phys.model_id(model);
}

//---------------------------------------------------------------------------//
/*!
 * Process interaction change.
 */
CELER_FUNCTION void post_process(GeoTrackView&      geo,
                                 ParticleTrackView& particle,
                                 PhysicsTrackView&  phys,
                                 SimTrackView&      sim,
                                 real_type*         edep,
                                 const Interaction& result)
{
    CELER_EXPECT(result);

    using Energy = ParticleTrackView::Energy;

    // Update the track state from the interaction
    // TODO: handle recoverable errors
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
    *edep += value_as<Energy>(result.energy_deposition);

    // Reset the physics state if a discrete interaction occured
    if (phys.model_id())
    {
        phys = {};
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
