//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsStepUtils.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"
#include "base/Algorithms.hh"
#include "base/NumericLimits.hh"
#include "base/Range.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate physics steps based on cross sections and range limits.
 */
CELER_FUNCTION real_type calc_tabulated_physics_step(
    const ParticleTrackView& particle, PhysicsTrackView& physics)
{
    CELER_EXPECT(physics.has_interaction_mfp());

    constexpr real_type inf = numeric_limits<real_type>::infinity();
    using VGT               = ValueGridType;

    // Loop over all processes that apply to this track (based on particle
    // type) and calculate cross section and particle range.
    real_type total_macro_xs = 0;
    real_type min_range      = inf;
    for (ParticleProcessId::value_type pp_idx :
         range(physics.num_particle_processes()))
    {
        const ParticleProcessId ppid{pp_idx};
        real_type               process_xs = 0;
        if (auto grid_id = physics.value_grid(VGT::macro_xs, ppid))
        {
            // Calculate macroscopic cross section for this process, then
            // accumulate it into the total cross section and save the cross
            // section for later.
            auto calc_xs = physics.make_calculator(grid_id);
            process_xs = calc_xs(particle.energy());
            total_macro_xs += process_xs;
        }
        else
        {
            // TODO: this won't work either because livermore does use
            // tables above 200 keV.
            ProcessId process = physics.process(ppid);
            if (process == physics.photoelectric_process_id())
            {
                CELER_NOT_IMPLEMENTED("on-the-fly photoelectric xs");
            }
            else if (process == physics.eplusgg_process_id())
            {
                CELER_NOT_IMPLEMENTED("on-the-fly positron annihilation xs");
            }
        }
        physics.per_process_xs(ppid) = process_xs;

        if (auto grid_id = physics.value_grid(VGT::range, ppid))
        {
            auto      calc_range    = physics.make_calculator(grid_id);
            real_type process_range = calc_range(particle.energy());
            // TODO: scale range by sqrt(particle.energy() / minKE)
            // if < minKE??
            min_range = min(min_range, process_range);
        }
    }

    if (min_range != inf)
    {
        // One or more range limiters applied: scale range limit according to
        // user options
        min_range = physics.range_to_step(min_range);
    }

    // Update step length with discrete interaction
    return min(min_range, physics.interaction_mfp() / total_macro_xs);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate energy loss over the given "true" step length.
 */
CELER_FUNCTION real_type calc_energy_loss(const ParticleTrackView& particle,
                                          const PhysicsTrackView&  physics,
                                          real_type                step)
{
    CELER_EXPECT(step >= 0);

    using VGT = ValueGridType;

    // Loop over all processes that apply to this track (based on particle
    // type) and calculate cross section and particle range.
    real_type total_eloss_rate = 0;
    for (ParticleProcessId::value_type pp_idx :
         range(physics.num_particle_processes()))
    {
        const ParticleProcessId ppid{pp_idx};
        if (auto grid_id = physics.value_grid(VGT::energy_loss, ppid))
        {
            auto calc_eloss_rate = physics.make_calculator(grid_id);
            total_eloss_rate += calc_eloss_rate(particle.energy());
        }
    }

    // TODO: reduce energy loss rate using range tables for individual
    // processes?? aka "long step" with max_eloss_fraction. Unlike Geant4 where
    // each process sequentially operates on the track, we can apply limits to
    // all energy loss processes simultaneouly.

    return total_eloss_rate * step;
}

//---------------------------------------------------------------------------//
/*!
 * Choose the physics model for a track's pending interaction.
 *
 * - If the interaction MFP is zero, the particle is undergoing a discrete
 *   interaction. Otherwise, the result is a false ModelId.
 * - Sample from the previously calculated per-process cross section/decay to
 *   determine the interacting process ID.
 * - From the process ID and (post-slowing-down) particle energy, we obtain the
 *   applicable model ID.
 */
template<class Engine>
CELER_FUNCTION ModelId select_model(const ParticleTrackView& particle,
                                    const PhysicsTrackView&  physics,
                                    Engine&                  rng)
{
    if (physics.interaction_mfp() > 0)
    {
        // Nonzero MFP to interaction -- no interaction model
        return {};
    }

    // Sample ParticleProcessId from physics.per_process_xs()
    (void)sizeof(rng);

    // Get ModelGroup from physics.models(ppid);

    // Find ModelId corresponding to energy bin
    (void)sizeof(particle.energy());

    CELER_NOT_IMPLEMENTED("selecting a model for interaction");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
