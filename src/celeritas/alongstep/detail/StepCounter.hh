//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/StepCounter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/global/CoreTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Increment the step counter and cut tracks that have taken too many steps.
 *
 * The result is whether the track is alive and has not been cut by this
 * function.
 */
struct StepCounter
{
    inline CELER_FUNCTION bool operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
CELER_FUNCTION bool StepCounter::operator()(CoreTrackView& track)
{
    auto sim = track.sim();
    CELER_EXPECT(sim.status() != TrackStatus::errored);

    sim.increment_num_steps();

    if (sim.status() != TrackStatus::alive)
    {
        return false;
    }

    CELER_ASSERT(sim.post_step_action());
    if (sim.num_steps() == sim.max_steps()
        && sim.post_step_action() != track.tracking_cut_action())
    {
#if !CELER_DEVICE_COMPILE
        CELER_LOG_LOCAL(debug) << "Track " << track.track_slot_id()
                               << " exceeded maximum step count";
#endif
        track.apply_cut();
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
