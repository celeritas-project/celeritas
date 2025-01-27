//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/ElossApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss using the EnergyLossHandler interface.
 *
 * TODO: move apply-cut out of mean/fluct eloss to this function to reduce
 * duplicate code?
 */
template<class EH>
struct ElossApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    EH eloss;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class EH>
CELER_FUNCTION ElossApplier(EH&&) -> ElossApplier<EH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class EH>
CELER_FUNCTION void ElossApplier<EH>::operator()(CoreTrackView const& track)
{
    auto particle = track.make_particle_view();
    if (!eloss.is_applicable(track) || particle.is_stopped())
    {
        return;
    }

    auto sim = track.make_sim_view();
    auto step = sim.step_length();
    auto post_step_action = sim.post_step_action();

    // Calculate energy loss, possibly applying tracking cuts
    bool apply_cut = (post_step_action != track.boundary_action());
    auto deposited = eloss.calc_eloss(track, step, apply_cut);
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
            >= track.make_physics_view().particle_scalars().lowest_energy
        || !apply_cut || particle.is_stopped());

    if (particle.is_stopped())
    {
        // Particle lost all energy over the step
        CELER_ASSERT(post_step_action != track.boundary_action());
        auto const phys = track.make_physics_view();
        if (!phys.has_at_rest())
        {
            // Immediately kill stopped particles with no at rest processes
            sim.status(TrackStatus::killed);
            sim.post_step_action(phys.scalars().range_action());
        }
        else
        {
            // Particle slowed down to zero: force a discrete interaction
            sim.post_step_action(phys.scalars().discrete_action());
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
