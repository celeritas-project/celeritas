//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/DiscreteSelectExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/track/SimTrackView.hh"

#include "../PhysicsData.hh"
#include "../PhysicsStepUtils.hh"
#include "../PhysicsTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
struct DiscreteSelectExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Select a physics process before undergoing a collision.
 */
CELER_FUNCTION void
DiscreteSelectExecutor::operator()(celeritas::CoreTrackView const& track)
{
    CELER_EXPECT(track.sim().status() == TrackStatus::alive);
    CELER_EXPECT(track.sim().post_step_action()
                 == track.physics().scalars().discrete_action());
    // Reset the MFP counter, to be resampled if the track survives the
    // interaction
    auto phys = track.physics();
    phys.reset_interaction_mfp();

    auto particle = track.particle();
    {
        // Select the action to take
        auto mat = track.material().material_record();
        auto rng = track.rng();
        auto step = track.physics_step();
        auto action
            = select_discrete_interaction(mat, particle, phys, step, rng);
        CELER_ASSERT(action);
        // Save it as the next kernel
        auto sim = track.sim();
        sim.post_step_action(action);
    }

    CELER_ENSURE(!phys.has_interaction_mfp());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
