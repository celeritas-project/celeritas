//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/TrackUpdater.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Finish the step.
 *
 * TODO: we may need to save the pre-step speed and apply the time update using
 * an average here.
 */
//---------------------------------------------------------------------------//
struct TrackUpdater
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION void TrackUpdater::operator()(CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (sim.status() != TrackStatus::killed)
    {
        CELER_ASSERT(sim.step_length() > 0
                     || track.make_particle_view().is_stopped());
        CELER_ASSERT(sim.post_step_action());
        auto phys = track.make_physics_view();
        if (sim.post_step_action() != phys.scalars().discrete_action())
        {
            // Reduce remaining mean free paths to travel. The 'discrete
            // action' case is launched separately and resets the
            // interaction MFP itself.
            auto step = track.make_physics_step_view();
            real_type mfp = phys.interaction_mfp()
                            - sim.step_length() * step.macro_xs();
            CELER_ASSERT(mfp > 0);
            phys.interaction_mfp(mfp);
        }
    }

    // Increment the step counter
    sim.increment_num_steps();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
