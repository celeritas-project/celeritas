//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/MeanELoss.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate energy loss (*without* fluctuations) to a track.
 */
class MeanELoss
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = ParticleTrackView::Energy;
    //!@}

  public:
    // Whether energy loss is used for this track
    inline CELER_FUNCTION bool is_applicable(CoreTrackView const&) const;

    // Apply to the track
    inline CELER_FUNCTION Energy calc_eloss(CoreTrackView const& track,
                                            real_type step,
                                            bool apply_cut);

    //! Particle will slow down to zero only if range limited
    static CELER_CONSTEXPR_FUNCTION bool imprecise_range() { return false; }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Whether energy loss is used for this track.
 */
CELER_FUNCTION bool MeanELoss::is_applicable(CoreTrackView const& track) const
{
    // The track can be marked as `errored` *within* the along-step kernel,
    // during propagation
    if (track.make_sim_view().status() == TrackStatus::errored)
        return false;

    // Energy loss grid ID is 'false'
    return static_cast<bool>(track.make_physics_view().energy_loss_grid());
}

//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 */
CELER_FUNCTION auto MeanELoss::calc_eloss(CoreTrackView const& track,
                                          real_type step,
                                          bool apply_cut) -> Energy
{
    CELER_EXPECT(step > 0);

    auto particle = track.make_particle_view();
    auto phys = track.make_physics_view();

    if (apply_cut && particle.energy() < phys.particle_scalars().lowest_energy)
    {
        // Deposit all energy when we start below the tracking cut
        return particle.energy();
    }

    // Calculate the mean energy loss
    Energy eloss = calc_mean_energy_loss(particle, phys, step);

    if (apply_cut
        && (particle.energy() - eloss <= phys.particle_scalars().lowest_energy))
    {
        // Deposit all energy when we end below the tracking cut
        return particle.energy();
    }

    CELER_ENSURE(eloss <= particle.energy());
    CELER_ENSURE(eloss != particle.energy()
                 || track.make_sim_view().post_step_action()
                        == phys.scalars().range_action());
    return eloss;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
