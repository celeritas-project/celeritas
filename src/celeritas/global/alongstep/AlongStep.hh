//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStep.hh
//! \brief Along-step function and helper classes
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "orange/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Perform the along-step action using helper functions.
 *
 * \tparam MH MSC helper, e.g. \c detail::NoMsc
 * \tparam MP Propagator factory, e.g. \c detail::LinearPropagatorFactory
 * \tparam EH Energy loss helper, e.g. \c detail::TrackNoEloss
 */
template<class MH, class MP, class EH>
inline CELER_FUNCTION void along_step(MH&& msc,
                                      MP&& make_propagator,
                                      EH&& eloss,
                                      CoreTrackView const& track)
{
    // TODO: scope the 'views' so that the lifetimes don't overlap between this
    // function and helper class functions
    auto sim = track.make_sim_view();

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
        CELER_ASSERT(track.make_physics_view().has_at_rest());
        // Increment the step counter
        sim.increment_num_steps();
        return;
    }

    bool use_msc = msc.is_applicable(track, step_limit.step);
    if (use_msc)
    {
        // Apply MSC step limiters and transform "physical" step (with MSC) to
        // "geometrical" step (smooth curve)
        msc.limit_step(track, &step_limit);
    }

    auto particle = track.make_particle_view();
    {
        auto geo = track.make_geo_view();
        auto propagate = make_propagator(particle, &geo);
        Propagation p = propagate(step_limit.step);
        if (propagate.tracks_can_loop())
        {
            sim.update_looping(p.looping);
        }
        if (propagate.tracks_can_loop() && p.looping)
        {
            // The track is looping, i.e. progressing little over many
            // integration steps in the field propagator (likely a low energy
            // particle in a low density material/strong magnetic field).
            step_limit.step = p.distance;
            step_limit.action = track.propagation_limit_action();

            // Kill the track if it's stable and below the threshold energy or
            // above the threshold number of steps allowed while looping.
            if (particle.is_stable()
                && sim.is_looping(particle.particle_id(), particle.energy()))
            {
                // If the track is looping (or if it's a stuck track that waa
                // flagged as looping), deposit the energy locally.
                auto deposited = particle.energy().value();
                if (particle.is_antiparticle())
                {
                    // Energy conservation for killed positrons
                    deposited += 2 * particle.mass().value();
                }
                track.make_physics_step_view().deposit_energy(
                    ParticleTrackView::Energy{deposited});
                particle.subtract_energy(particle.energy());

                // Mark that this track was abandoned while looping
                step_limit.action = track.abandon_looping_action();
                sim.force_step_limit(step_limit);
                sim.increment_num_steps();
                sim.status(TrackStatus::killed);
                return;
            }
        }
        else
        {
            if (p.boundary)
            {
                // Stopped at a geometry boundary: this is the new step action.
                CELER_ASSERT(p.distance <= step_limit.step);
                step_limit.step = p.distance;
                step_limit.action = track.boundary_action();
            }
            else if (p.distance < step_limit.step)
            {
                // Some tracks may get stuck on a boundary and fail to move at
                // all in the field propagator, and will get bumped a small
                // distance. This primarily occurs with reentrant tracks on a
                // boundary with VecGeom.
                step_limit.step = p.distance;
                step_limit.action = track.propagation_limit_action();
            }
        }
    }

    if (use_msc)
    {
        // Scatter the track and transform the "geometrical" step back to
        // "physical" step
        msc.apply_step(track, &step_limit);
    }

    // Update track's lab-frame time using the beginning-of-step speed
    {
        CELER_ASSERT(!particle.is_stopped());
        real_type speed = native_value_from(particle.speed());
        CELER_ASSERT(speed >= 0);
        if (speed > 0)
        {
            // For very small energies (< numeric_limits<real_type>::epsilon)
            // the calculated speed can be zero.
            real_type delta_time = step_limit.step / speed;
            sim.add_time(delta_time);
        }
    }

    if (eloss.is_applicable(track))
    {
        using Energy = ParticleTrackView::Energy;

        bool apply_cut = (step_limit.action != track.boundary_action());
        Energy deposited = eloss.calc_eloss(track, step_limit.step, apply_cut);
        CELER_ASSERT(deposited <= particle.energy());
        CELER_ASSERT(apply_cut || deposited != particle.energy());

        if (deposited > zero_quantity())
        {
            // Deposit energy loss
            auto step = track.make_physics_step_view();
            step.deposit_energy(deposited);
            particle.subtract_energy(deposited);
        }
        // Energy loss helper *must* apply the tracking cutoff
        CELER_ASSERT(
            particle.energy()
                >= track.make_physics_view().scalars().lowest_electron_energy
            || !apply_cut || particle.is_stopped());
    }

    if (particle.is_stopped())
    {
        // Particle lost all energy over the step
        CELER_ASSERT(step_limit.action != track.boundary_action());
        auto phys = track.make_physics_view();
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

    if (sim.status() != TrackStatus::killed)
    {
        CELER_ASSERT(step_limit.step > 0);
        CELER_ASSERT(step_limit.action);
        auto phys = track.make_physics_view();
        if (step_limit.action != phys.scalars().discrete_action())
        {
            // Reduce remaining mean free paths to travel. The 'discrete
            // action' case is launched separately and resets the
            // interaction MFP itself.
            auto step = track.make_physics_step_view();
            real_type mfp = phys.interaction_mfp()
                            - step_limit.step * step.macro_xs();
            CELER_ASSERT(mfp > 0);
            phys.interaction_mfp(mfp);
        }
    }

    {
        // Override step limit with action/step changes we applied
        sim.force_step_limit(step_limit);
        // Increment the step counter
        sim.increment_num_steps();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
