//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/DiscreteSelectExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

#include "../CoreTrackView.hh"
#include "../PhysicsStepUtils.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 */
struct DiscreteSelectExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Select a physics model before undergoing a discrete interaction.
 */
CELER_FUNCTION void
DiscreteSelectExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(track.make_sim_view().status() == TrackStatus::alive);
    CELER_EXPECT(track.make_sim_view().post_step_action()
                 == track.make_physics_view().scalars().discrete_action());

    // Reset the MFP counter, to be resampled if the track survives the
    // interaction
    auto phys = track.make_physics_view();
    phys.reset_interaction_mfp();

    auto particle = track.make_particle_view();
    auto step = track.make_physics_step_view();
    auto rng = track.make_rng_engine();

    auto action = select_discrete_interaction(phys, step, rng);

    CELER_ASSERT(action);

    // Save the action as the next kernel
    auto sim = track.make_sim_view();
    sim.post_step_action(action);

    CELER_ENSURE(!phys.has_interaction_mfp());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
