//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
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
                                     Rng&                     rng)
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
 * Propagate up to the step length or next boundary.
 */
CELER_FUNCTION real_type propagate(GeoTrackView&           geo,
                                   const PhysicsTrackView& phys)
{
    // Actual distance, limited by along-step length or geometry
    real_type step = phys.step_length();
    if (step > 0)
    {
        // Propagate up to the step length or next boundary
        LinearPropagator propagate(&geo);
        auto             geo_step = propagate(step);
        step                      = geo_step.distance;
        // TODO: check whether the volume/material have changed?
    }
    return step;
}

//---------------------------------------------------------------------------//
/*!
 * Select the model for the discrete interaction.
 */
template<class Rng>
CELER_FUNCTION void select_discrete_model(ParticleTrackView&        particle,
                                          PhysicsTrackView&         phys,
                                          Rng&                      rng,
                                          real_type                 step,
                                          ParticleTrackView::Energy eloss)
{
    // Reduce the energy and path length
    particle.energy(
        ParticleTrackView::Energy{particle.energy().value() - eloss.value()});
    phys.step_length(phys.step_length() - step);

    // Reduce the remaining mean free path
    real_type mfp = phys.interaction_mfp() - step * phys.macro_xs();
    phys.interaction_mfp(soft_zero(mfp) ? 0 : mfp);

    // Reached the interaction point: sample the process and determine the
    // corresponding model
    ModelId model{};
    if (phys.interaction_mfp() <= 0)
    {
        auto ppid_mid = select_process_and_model(particle, phys, rng);
        model         = ppid_mid.model;
    }
    phys.model_id(model);
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
