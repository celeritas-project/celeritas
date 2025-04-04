//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DiscreteSelectExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/random/RngEngine.hh"

#include "../CoreTrackView.hh"
#include "../PhysicsData.hh"
#include "../PhysicsStepUtils.hh"
#include "../PhysicsTrackView.hh"
#include "../SimTrackView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct DiscreteSelectExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Select a physics process before undergoing a collision.
 */
CELER_FUNCTION void
DiscreteSelectExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(track.sim().status() == TrackStatus::alive);
    CELER_EXPECT(track.sim().post_step_action()
                 == track.physics().discrete_action());

    // Reset the MFP counter, to be resampled if the track survives the
    // interaction
    auto phys = track.physics();
    phys.reset_interaction_mfp();

    // Select the action to take
    auto particle = track.particle();
    auto rng = track.rng();
    auto action = select_discrete_interaction(particle, phys, rng);
    CELER_ASSERT(action);

    // Save action as next kernel
    auto sim = track.sim();
    sim.post_step_action(action);

    CELER_ENSURE(!phys.has_interaction_mfp());
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
