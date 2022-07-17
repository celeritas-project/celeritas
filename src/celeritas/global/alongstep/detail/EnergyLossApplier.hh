//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/EnergyLossApplier.hh
//---------------------------------------------------------------------------//
#pragma once

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

    // Calculate mean energy loss
    auto   particle = track.make_particle_view();
    Energy eloss    = calc_mean_energy_loss(particle, phys, step_limit->step);
    if (eloss > zero_quantity())
    {
        // Deposit energy loss
        auto step = track.make_physics_step_view();
        step.deposit_energy(eloss);
        particle.subtract_energy(eloss);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
