//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/ElossApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

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
    auto particle = track.particle();
    auto sim = track.sim();
    auto phys = track.physics();

    if (sim.status() == TrackStatus::errored)
    {
        // Failed during propagation
        return;
    }
    if (!phys.energy_loss_grid())
    {
        // No energy loss for this particle/material
        return;
    }
    if (particle.is_stopped())
    {
        // No energy to lose
        return;
    }
    CELER_ASSERT(sim.step_length() > 0);

    // Avoid stopping particles unphysically on the boundary: particles should
    // theoretically only slow to zero via range action, but spline
    // interpolation and energy fluctuations are inconsistent and may lead to
    // incorrectly long steps
    bool const on_boundary = track.geometry().is_on_boundary();
    CELER_ASSERT(on_boundary
                 == (sim.post_step_action() == track.boundary_action()));

    ParticleTrackView::Energy deposited;
    if (!on_boundary
        && particle.energy() < phys.particle_scalars().lowest_energy)
    {
        // Beginning-of-step energy is below the tracking cut: deposit all
        // remaining energy along the step
        deposited = particle.energy();
    }
    else
    {
        // Calculate energy loss along the step
        deposited = eloss.calc_eloss(track);
    }

    // Calculate energy loss, possibly applying tracking cuts
    auto pstep = track.physics_step();
    apply_slowing_down(
        track.physics(), on_boundary, deposited, particle, pstep, sim);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
