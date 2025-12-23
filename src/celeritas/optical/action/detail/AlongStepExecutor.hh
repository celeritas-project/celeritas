//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/AlongStepExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Complete end-of-step activity for a track.
 *
 * - Update track time
 * - Update number of steps
 * - Update remaining MFPs to interaction
 */
struct AlongStepExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void AlongStepExecutor::operator()(CoreTrackView& track)
{
    auto sim = track.sim();

    CELER_ASSERT(sim.status() == TrackStatus::alive);
    CELER_ASSERT(sim.step_length() > 0);
    CELER_ASSERT(sim.post_step_action());

    // Update time
    sim.add_time(sim.step_length() / constants::c_light);

    // Increment the step counter
    sim.increment_num_steps();

    // Kill the track if it's reached the step limit
    if (sim.num_steps() == sim.max_steps())
    {
#if !CELER_DEVICE_COMPILE
        CELER_LOG_LOCAL(error) << R"(Track exceeded maximum step count)";
#endif
        track.apply_errored();
        return;
    }

    // Update remaining MFPs to interaction
    auto phys = track.physics();
    if (sim.post_step_action() != phys.discrete_action())
    {
        // Reduce remaining mean free paths to travel. The 'discrete action'
        // case is launched separately and resets the interaction MFP itself.
        real_type mfp = phys.interaction_mfp()
                        - sim.step_length() * phys.macro_xs();
        CELER_ASSERT(mfp > 0);
        phys.interaction_mfp(mfp);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
