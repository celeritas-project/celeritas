//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/PropagateExecutor.hh
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
 * Move a track to the next interaction or geometry boundary.
 *
 * This should only apply to alive tracks.
 */
struct PropagateExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
CELER_FUNCTION void PropagateExecutor::operator()(CoreTrackView const& track)
{
    auto sim = track.sim();
    CELER_ASSERT(sim.status() == TrackStatus::alive);

    // Propagate
    real_type step = sim.step_length();
    CELER_ASSERT(step > 0);
    Propagation p = geo_.find_next_step(step);
    if (p.boundary)
    {
        geo_.move_to_boundary();
        step = p.distance;
        sim.step_length(p.distance);
        sim.post_step_action(track.boundary_action());
    }
    else
    {
        CELER_ASSERT(step == result.distance);
        geo_.move_internal(step);
    }

    // Update time
    sim.add_time(step / constants::c_light);

    // Update track
    sim.increment_num_steps();

    CELER_ASSERT(sim.status() == TrackStatus::alive);
    CELER_ASSERT(sim.step_length() > 0);
    CELER_ASSERT(sim.post_step_action());

    if (sim.num_steps() == sim.max_steps()
        && sim.post_step_action() != track.tracking_cut_action())
    {
#if !CELER_DEVICE_COMPILE
        CELER_LOG_LOCAL(error) << R"(Track exceeded maximum step count)";
#endif
        track.apply_errored();
        return;
    }
    else
    {
        // TODO: reduce MFP by step * xs
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
