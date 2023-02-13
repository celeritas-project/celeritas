//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/MeanELoss.hh
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
struct MeanELoss
{
    //!@{
    //! \name Type aliases
    using Energy = ParticleTrackView::Energy;
    //!@}

    //// MEMBER FUNCTIONS ////

    //! Particle will slow down to zero only if range limited
    static CELER_CONSTEXPR_FUNCTION bool imprecise_range() { return false; }

    // Whether energy loss is used for this track
    CELER_FUNCTION bool is_applicable(CoreTrackView const&) const;

    // Apply to the track
    inline CELER_FUNCTION Energy calc_eloss(CoreTrackView const& track,
                                            real_type step);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Whether energy loss is used for this track.
 */
CELER_FUNCTION bool MeanELoss::is_applicable(CoreTrackView const& track) const
{
    // Energy loss grid ID will be 'false' if inapplicable
    auto ppid = track.make_physics_view().eloss_ppid();
    return static_cast<bool>(ppid);
}

//---------------------------------------------------------------------------//
/*!
 * Apply energy loss to the given track.
 */
CELER_FUNCTION auto
MeanELoss::calc_eloss(CoreTrackView const& track, real_type step) -> Energy
{
    CELER_EXPECT(step > 0);

    // Calculate the true energy loss
    auto particle = track.make_particle_view();
    auto phys = track.make_physics_view();
    return calc_mean_energy_loss(particle, phys, step);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
