//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/EnergyLossApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss (*without* fluctuations) to a track.
 */
struct EnergyLossApplier
{
    //!@{
    //! \name Type aliases
    using Energy = ParticleTrackView::Energy;
    //!@}

    //// MEMBER FUNCTIONS ////

    // Apply to the track
    inline CELER_FUNCTION void
    operator()(CoreTrackView const& track, StepLimit* step_limit);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 */
CELER_FUNCTION void EnergyLossApplier::operator()(CoreTrackView const& track,
                                                  StepLimit* step_limit)
{
    CELER_EXPECT(step_limit->step > 0);

    auto phys = track.make_physics_view();
    auto ppid = phys.eloss_ppid();
    if (!ppid)
    {
        // No energy loss process for this particle type
        return;
    }

    auto particle = track.make_particle_view();
    Energy eloss;
    if (particle.energy() < phys.scalars().eloss_calc_limit
        && step_limit->action != track.boundary_action())
    {
        // Immediately stop low-energy tracks (as long as they're not crossing
        // a boundary)
        // TODO: this should happen before creating tracks from secondaries
        // *OR* after slowing down tracks: duplicated in
        // EnergyLossFluctApplier.hh
        eloss = particle.energy();
    }
    else
    {
        // Calculate mean energy loss
        eloss = calc_mean_energy_loss(particle, phys, step_limit->step);
    }

    CELER_ASSERT(eloss <= particle.energy());
    if (eloss == particle.energy())
    {
        // Particle lost all energy over the step: this can happen if we're
        // range limited *or* if below the hard cutoff
        CELER_ASSERT(step_limit->action != track.boundary_action());

        if (!phys.has_at_rest())
        {
            // Immediately kill stopped particles with no at rest processes
            auto sim = track.make_sim_view();
            sim.status(TrackStatus::killed);
        }
        else
        {
            // Particle slowed down to zero: force a discrete interaction
            step_limit->action = phys.scalars().discrete_action();
        }
    }

    if (eloss > zero_quantity())
    {
        // Deposit energy loss
        auto step = track.make_physics_step_view();
        step.deposit_energy(eloss);
        particle.subtract_energy(eloss);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
