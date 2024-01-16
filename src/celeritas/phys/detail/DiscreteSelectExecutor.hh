//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/phys/PhysicsData.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/track/SimTrackView.hh"

#include "../PhysicsStepUtils.hh"

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
    CELER_EXPECT(track.make_sim_view().status() == TrackStatus::alive);
    CELER_EXPECT(track.make_sim_view().post_step_action()
                 == track.make_physics_view().scalars().discrete_action());
    // Reset the MFP counter, to be resampled if the track survives the
    // interaction
    auto phys = track.make_physics_view();
    phys.reset_interaction_mfp();

    auto particle = track.make_particle_view();
    {
        // Select the action to take
        auto mat = track.make_material_view().make_material_view();
        auto rng = track.make_rng_engine();
        auto step = track.make_physics_step_view();
        auto action
            = select_discrete_interaction(mat, particle, phys, step, rng);
        CELER_ASSERT(action);
        // Save it as the next kernel
        auto sim = track.make_sim_view();
        sim.post_step_action(action);
    }

    CELER_ENSURE(!phys.has_interaction_mfp());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
