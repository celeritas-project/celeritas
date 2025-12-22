//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/ElossApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/detail/QuantityImpl.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Whether the given track can lose energy along its step.
 */
inline bool is_eloss_applicable(CoreTrackView const& track)
{
    auto sim = track.sim();
    if (sim.status() != TrackStatus::alive)
    {
        // Errored during propagation
        return false;
    }

    if (track.particle().is_stopped())
    {
        // No energy to lose
        return false;
    }

    if (!track.physics().energy_loss_grid())
    {
        // No energy loss for this particle/material
        return false;
    }

    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Deposit energy along the particle's step and update the particle state.
 *
 * - Particles that end below the tracking cut distribute their remaining
 *   energy along the step, unless they end on the boundary
 * - Energy loss is removed to the particle and added to the physics step
 * - Stopped tracks are killed if they have no at-rest process
 * - Stopped tracks with at-rest processes are forced to undergo an interaction
 */
inline CELER_FUNCTION void
apply_slowing_down(CoreTrackView const& track, ParticleTrackView::Energy eloss)
{
    auto particle = track.particle();
    auto phys = track.physics();
    auto sim = track.sim();
    bool on_boundary = track.geometry().is_on_boundary();

    CELER_EXPECT(eloss > zero_quantity());
    CELER_EXPECT(eloss <= particle.energy());
    CELER_EXPECT(eloss != particle.energy() || !on_boundary
                 || sim.post_step_action() == phys.scalars().range_action());

    if (!on_boundary
        && (particle.energy() - eloss <= phys.particle_scalars().lowest_energy))
    {
        // Particle ended below the tracking cut: deposit all its energy
        // (aka adjusting dE/dx upward a bit)
        eloss = particle.energy();
    }
    if (eloss > zero_quantity())
    {
        // Deposit energy loss
        track.physics_step().deposit_energy_from(eloss, particle);
    }

    // At this point, we shouldn't have any low-energy tracks *except* on the
    // boundary
    CELER_ASSERT(particle.energy() >= phys.particle_scalars().lowest_energy
                 || on_boundary || particle.is_stopped());

    if (particle.is_stopped())
    {
        if (!phys.at_rest_process())
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
/*!
 * Apply energy loss using an energy loss calculator class.
 *
 * \todo Rename to \c ElossExecutor.
 */
template<class EC>
struct ElossApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    EC calc_eloss;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class EC>
CELER_FUNCTION void ElossApplier<EC>::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(track.sim().step_length() > 0);
    if (!is_eloss_applicable(track))
        return;

    using Energy = ParticleTrackView::Energy;
    Energy deposited = [this, &track]() -> Energy {
        // Avoid stopping particles unphysically on the boundary: particles
        // should theoretically only slow to zero via range action, but spline
        // interpolation and energy fluctuations are inconsistent and may lead
        // to incorrectly long steps
        bool const on_boundary = track.geometry().is_on_boundary();
        auto sim = track.sim();
        CELER_ASSERT(on_boundary
                     == (sim.post_step_action() == track.boundary_action()));

        auto particle = track.particle();
        auto phys = track.physics();
        if (!on_boundary
            && particle.energy() < phys.particle_scalars().lowest_energy)
        {
            // Beginning-of-step energy is below the tracking cut: deposit all
            // remaining energy along the step
            return particle.energy();
        }
        // Calculate energy loss along the step
        return this->calc_eloss(track);
    }();

    if (deposited > zero_quantity())
    {
        // Apply
        apply_slowing_down(track, deposited);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
