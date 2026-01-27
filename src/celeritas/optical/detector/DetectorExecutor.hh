//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
struct DetectorExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void
DetectorExecutor::operator()(CoreTrackView const& track) const
{
    auto score = track.scoring();
    auto sim = track.sim();

    if (sim.status() == TrackStatus::alive)
    {
        auto const detectors = track.detectors();

        auto geometry = track.geometry();

        auto const volume_id = geometry.volume_id();
        auto const detector_id = detectors.detector_id(volume_id);

        if (detector_id)
        {
            score.score_hit(DetectorHit{detector_id,
                                        track.particle().energy(),
                                        sim.time(),
                                        geometry.pos(),
                                        geometry.volume_instance_id()});

            // Kill the track
            sim.status(TrackStatus::killed);
        }
        else
        {
            score.clear_hit();
        }
    }
    else
    {
        // Ensure killed, inactive, and errored tracks don't contribute to hits
        score.clear_hit();
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
