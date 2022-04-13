//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AlongStepActionImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "geometry/LinearPropagator.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsStepUtils.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "physics/em/detail/UrbanMscData.hh"
#include "physics/em/detail/UrbanMscScatter.hh"
#include "physics/em/detail/UrbanMscStepLimit.hh"
#include "sim/CoreTrackData.hh"
#include "sim/CoreTrackView.hh"

namespace celeritas
{
namespace detail
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
/*!
 * Move to the step endpoint.
 *
 * - Determine step limitation for multiple scattering, if applicable
 * - Move within the geometry to the end of the current limited step or the
 *   geometry boundary
 * - Apply multiple scattering displacment/direction change, if applicable
 * - Calculate and deposit energy loss along the step
 *
 * TODO:
 * - change to launcher class that can be templated on propagator, MSC type
 */
inline CELER_FUNCTION void
along_step_track(celeritas::CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (sim.status() == TrackStatus::inactive)
    {
        // Track slot is empty
        CELER_ASSERT(!sim.step_limit());
        return;
    }

    // Increment the step counter
    sim.increment_num_steps();

    // True step is the actual path length traveled by the particle, including
    // within-step MSC
    StepLimit step_limit = sim.step_limit();
    CELER_ASSERT(step_limit);
    if (step_limit.step == 0)
    {
        // Track is stopped: no movement or energy loss will happen
        // (could be a stopped positron waiting for annihilation, or a particle
        // waiting to decay?)
        CELER_ASSERT(track.make_particle_view().is_stopped());
        CELER_ASSERT(step_limit.action
                     == track.make_physics_view().scalars().discrete_action());
        return;
    }

    auto particle = track.make_particle_view();
    auto phys     = track.make_physics_view();
    auto geo      = track.make_geo_view();

    // Geometry step is the continuous-line movement (straight if no magnetic
    // field, curved if) of the track. It will equal the true step if no MSC is
    // in use.
    real_type geo_step = step_limit.step;
    bool      use_msc  = use_msc_track(particle, phys, geo_step);
    if (use_msc)
    {
        auto mat = track.make_material_view();
        auto rng = track.make_rng_engine();
        // Sample multiple scattering step length
        celeritas::detail::UrbanMscStepLimit msc_step_limit(
            phys.urban_data(),
            particle,
            &geo,
            phys,
            mat.make_material_view(),
            sim,
            step_limit.step);

        auto msc_step_result = msc_step_limit(rng);
        phys.msc_step(msc_step_result);

        // Use "straight line" path calculated for geometry step
        geo_step = msc_step_result.geom_path;

        if (msc_step_result.true_path < step_limit.step)
        {
            // True/physical step might be further limited by MSC
            // TODO: this is already kinda sorta determined inside the
            // UrbanMscStepLimit calculation
            step_limit.step   = msc_step_result.true_path;
            step_limit.action = phys.urban_data().ids.action;
        }
    }

    {
        // Propagate up to the geometric step length
        LinearPropagator propagate(&geo);
        auto             propagated = propagate(geo_step);
        if (propagated.boundary)
        {
            // Stopped at a geometry boundary: this is the new step action. The
            // distance might *not* be correct if MSC is being used.
            geo_step          = propagated.distance;
            step_limit.action = track.boundary_action();
        }
    }

    // Calculate energy loss over the step length
    auto mat     = track.make_material_view();
    auto rng     = track.make_rng_engine();
    auto cutoffs = track.make_cutoff_view();

    // Sample the multiple scattering
    if (use_msc)
    {
        const auto& urban_data = phys.urban_data();

        // Replace step with actual geometry distance traveled
        auto msc_step_result      = phys.msc_step();
        msc_step_result.geom_path = geo_step;

        celeritas::detail::UrbanMscScatter msc_scatter(urban_data,
                                                       particle,
                                                       &geo,
                                                       phys,
                                                       mat.make_material_view(),
                                                       msc_step_result);
        auto msc_result = msc_scatter(rng);

        // Update full path length traveled along the step based on MSC to
        // correctly calculate energy loss, step time, etc.
        CELER_ASSERT(geo_step <= msc_result.step_length
                     && msc_result.step_length <= step_limit.step);
        step_limit.step = msc_result.step_length;

        // Update direction and position
        geo.set_dir(msc_result.direction);
        celeritas::Real3 new_pos;
        for (int i : celeritas::range(3))
        {
            new_pos[i] = geo.pos()[i] + msc_result.displacement[i];
        }
        geo.move_internal(new_pos);
    }
    else
    {
        // Step might have changed due to geometry boundary
        step_limit.step = geo_step;
    }

    // TODO: update track's lab-frame time here

    using Energy = ParticleTrackView::Energy;
    Energy eloss
        = calc_energy_loss(cutoffs, mat, particle, phys, step_limit.step, rng);
    if (eloss == particle.energy())
    {
        // Particle lost all energy over the step
        if (CELER_UNLIKELY(step_limit.action != phys.scalars().range_action()
                           && !use_msc))
        {
            // Particle *was* not range-limited. This means it physically moved
            // too far over the step, since the range is supposed to be the
            // integral of the inverse energy loss rate. Back particle slightly
            // away from boundary to avoid on-surface initialization/direction
            // change.
            // NOTE: this treatment cannot be used if MSC or magnetic fields
            // are active, because the direction has changed since the start of
            // the step!
            real_type backward_bump = real_type(-1e-5) * step_limit.step;
            // Force the step limiter to be "range"
            step_limit.action = phys.scalars().range_action();
            step_limit.step += backward_bump;

            celeritas::Real3 pos = geo.pos();
            axpy(backward_bump, geo.dir(), &pos);
            geo.move_internal(pos);
        }

        if (!phys.has_at_rest())
        {
            // Immediately kill stopped particles with no at rest processes
            sim.status(TrackStatus::killed);
        }
        else
        {
            // Particle slowed down to zero: force a discrete interaction
            step_limit.action = phys.scalars().discrete_action();
        }
    }

    // Deposit energy loss
    phys.deposit_energy(eloss);
    particle.subtract_energy(eloss);

    if (step_limit.action != phys.scalars().discrete_action())
    {
        // Reduce remaining mean free paths to travel. The 'discrete action'
        // case is launched separately and resets the interaction MFP itself.
        real_type mfp = phys.interaction_mfp()
                        - step_limit.step * phys.macro_xs();
        CELER_ASSERT(mfp > 0);
        phys.interaction_mfp(mfp);
    }

    // Override step limit with whatever action/step changes we applied here
    sim.force_step_limit(step_limit);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
