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
    inline CELER_FUNCTION Energy operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 */
CELER_FUNCTION auto MeanELoss::operator()(CoreTrackView const& track) -> Energy
{
    auto particle = track.particle();
    auto phys = track.physics();
    auto sim = track.sim();
    CELER_EXPECT(!particle.is_stopped() && sim.step_length() > 0);

    // Calculate the mean energy loss
    return calc_mean_energy_loss(particle, phys, sim.step_length());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
