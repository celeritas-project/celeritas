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
#include "geometry/LinearPropagator.hh"
#include "physics/base/PhysicsStepUtils.hh"
#include "physics/em/detail/UrbanMscData.hh"
#include "physics/em/detail/UrbanMscScatter.hh"
#include "physics/em/detail/UrbanMscStepLimit.hh"
#include "random/distributions/ExponentialDistribution.hh"
#include "sim/CoreTrackData.hh"
#include "sim/CoreTrackView.hh"

#ifndef CELER_DEVICE_COMPILE
#    include "base/ArrayIO.hh"
#    include "comm/Logger.hh"
#endif

using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::value_as;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
inline CELER_FUNCTION bool
use_msc_track(const celeritas::ParticleTrackView& particle,
              const celeritas::PhysicsTrackView&  phys,
              celeritas::real_type                step)
{
    if (!phys.msc_ppid())
        return false;
    const auto& urban_data = phys.urban_data();
    return (step > urban_data.params.geom_limit
            && particle.energy() > urban_data.params.energy_limit);
}
} // namespace

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Pre-step logic.
 *
 * Sample the mean free path and calculate the physics step limits.
 */
inline CELER_FUNCTION void pre_step_track(celeritas::CoreTrackView const& track)
{
    using celeritas::Action;
    using celeritas::ExponentialDistribution;
    using celeritas::real_type;

    // Clear out energy deposition
    track.energy_deposition() = 0;

    auto sim = track.make_sim_view();
    if (!sim.alive())
        return;

    auto phys = track.make_physics_view();

    // Sample mean free path
    if (!phys.has_interaction_mfp())
    {
        auto                               rng = track.make_rng_engine();
        ExponentialDistribution<real_type> sample_exponential;
        phys.interaction_mfp(sample_exponential(rng));
    }

    // Calculate physics step limits and total macro xs
    auto      mat      = track.make_material_view();
    auto      particle = track.make_particle_view();
    real_type step     = calc_tabulated_physics_step(mat, particle, phys);
    if (particle.is_stopped())
    {
        if (phys.macro_xs() == 0)
        {
            // If the particle is stopped and cannot undergo a discrete
            // interaction, kill it
            track.interaction().action = Action::cutoff_energy;
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
inline CELER_FUNCTION void
along_and_post_step_track(celeritas::CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (!sim.alive())
    {
        // Clear the model ID so inactive tracks will exit the interaction
        // kernels
        track.reset_model_id();
        return;
    }

    using Energy           = celeritas::ParticleTrackView::Energy;
    using Action           = celeritas::Action;
    using real_type        = celeritas::real_type;
    using LinearPropagator = celeritas::LinearPropagator;
    using ModelId          = celeritas::ModelId;

    auto      phys             = track.make_physics_view();
    auto      particle         = track.make_particle_view();
    bool      crossed_boundary = false;
    real_type step             = phys.step_length();
    real_type start_energy     = value_as<Energy>(particle.energy());

    Action result_action = Action::unchanged;

    if (!particle.is_stopped())
    {
        auto geo = track.make_geo_view();
        auto mat = track.make_material_view();
        auto rng = track.make_rng_engine();
        bool use_msc = use_msc_track(particle, phys, step);

        if (use_msc)
        {
            // Sample multiple scattering step length
            celeritas::detail::UrbanMscStepLimit msc_step_limit(
                phys.urban_data(),
                particle,
                &geo,
                phys,
                mat.make_material_view(),
                sim);
            auto msc_step_result = msc_step_limit(rng);

            // Limit geometry step
            step = msc_step_result.geom_path;

            // Test for the msc limited step
            if (msc_step_result.true_path < msc_step_result.phys_step)
            {
                result_action = Action::msc_limited;
            }
            phys.msc_step(msc_step_result);
        }

        // Boundary crossings and energy loss only need to be considered when
        // the particle is moving
        CELER_ASSERT(step > 0);
        {
            // Propagate up to the step length or next boundary
            LinearPropagator propagate(&geo);

            auto geo_step    = propagate(step);
            step             = geo_step.distance;
            crossed_boundary = geo_step.boundary;
        }

        // Sample the multiple scattering
        if (use_msc)
        {
            const auto& urban_data = phys.urban_data();

            // Replace step with actual geometry distance traveled
            auto step = phys.msc_step();
            step.geom_path = step;

            celeritas::detail::UrbanMscScatter msc_scatter(
                urban_data,
                particle,
                &geo,
                phys,
                mat.make_material_view(), step);
            auto msc_result = msc_scatter(rng);
            // Restore full path length traveled along the step to
            // correctly calculate energy loss, step time, etc.
            step = msc_result.step_length;

            // Update direction and position
            geo.set_dir(msc_result.direction);
            celeritas::Real3 new_pos = geo.pos();
            celeritas::axpy(real_type(1), msc_result.displacement, &new_pos);
            geo.move_internal(new_pos);
        }

        // Calculate energy loss over the step length
        auto      cutoffs = track.make_cutoff_view();
        real_type eloss   = value_as<Energy>(
            calc_energy_loss(cutoffs, mat, particle, phys, step, rng));
        track.energy_deposition() += eloss;
        particle.energy(Energy{start_energy - eloss});
    }

    ModelId result_model{};

    if (particle.is_stopped())
    {
        if (CELER_UNLIKELY(crossed_boundary))
        {
            auto geo = track.make_geo_view();
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
        auto geo = track.make_geo_view();
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
            auto geo_mat  = track.make_geo_material_view();
            auto matid    = geo_mat.material_id(geo.volume_id());
            CELER_ASSERT(matid);
            auto mat = track.make_material_view();
            mat      = {matid};
        }
    }
    else if (phys.interaction_mfp() <= 0)
    {
        // Reached the interaction point: sample the process and determine
        // the corresponding model
        auto rng      = track.make_rng_engine();
        auto ppid_mid = select_process_and_model(particle, phys, rng);
        result_model  = ppid_mid.model;
    }
    phys.model_id(result_model);
    track.interaction().action = result_action;
    track.step_length()        = step;
}

//---------------------------------------------------------------------------//
/*!
 * Postprocess secondaries and interaction results.
 */
inline CELER_FUNCTION void
process_interactions_track(celeritas::CoreTrackView const& track)
{
    using Energy = celeritas::ParticleTrackView::Energy;

    auto sim = track.make_sim_view();

    // Increment the step count before checking if the track is alive as some
    // active tracks might have been killed earlier in the step.
    sim.increment_num_steps();

    if (!sim.alive() || action_msc(track.interaction().action))
    {
        return;
    }

    // Update the track state from the interaction
    // TODO: handle recoverable errors
    const celeritas::Interaction& result = track.interaction();
    CELER_ASSERT(result);
    if (action_killed(result.action))
    {
        sim.alive(false);
    }
    else if (!action_unchanged(result.action))
    {
        auto particle = track.make_particle_view();
        particle.energy(result.energy);

        auto geo = track.make_geo_view();
        geo.set_dir(result.direction);
    }

    // Deposit energy from interaction
    track.energy_deposition() += value_as<Energy>(result.energy_deposition);
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries.
 */
inline CELER_FUNCTION void cleanup_track(celeritas::CoreTrackView const& track)
{
    CELER_ASSERT(track.thread_id().get() == 0);

    auto alloc = track.make_secondary_allocator();
    alloc.clear();
}

//---------------------------------------------------------------------------//
} // namespace demo_loop
