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
    // Apply to the track
    inline CELER_FUNCTION Energy calc_eloss(CoreTrackView const& track,
                                            real_type step,
                                            bool apply_cut);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 */
CELER_FUNCTION auto MeanELoss::calc_eloss(CoreTrackView const& track,
                                          real_type step,
                                          bool apply_cut) -> Energy
{
    CELER_EXPECT(step > 0);

    auto particle = track.particle();
    auto phys = track.physics();

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
                 || track.sim().post_step_action()
                        == phys.scalars().range_action());
    return eloss;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
