//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/detail/DetectorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/DetectorData.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Populate detector state buffer at the end of a step.
 *
 * All tracks have hits copied into the state buffer. If the track is not alive
 * or is not in a detector region, an invalid hit is set in the corresponding
 * buffer track slot.
 *
 * When a track generates a valid hit, it is killed (absorbed by the detector).
 */
struct DetectorExecutor
{
    NativeRef<DetectorStateData> detector_state_;

    // Copy track hit into the state buffer
    inline CELER_FUNCTION void operator()(CoreTrackView const&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Copy track hit into the state buffer.
 */
CELER_FUNCTION void
DetectorExecutor::operator()(CoreTrackView const& track) const
{
    auto& hit = detector_state_.detector_hits[track.track_slot_id()];
    auto sim = track.sim();

    if (sim.status() == TrackStatus::alive)
    {
        auto const detectors = track.detectors();

        auto geometry = track.geometry();

        auto const volume_id = geometry.volume_id();
        auto const detector_id = detectors.detector_id(volume_id);

        if (detector_id)
        {
            // Score a valid hit
            hit = DetectorHit{detector_id,
                              track.particle().energy(),
                              sim.time(),
                              geometry.pos(),
                              geometry.volume_instance_id()};

            // Kill the track
            sim.status(TrackStatus::killed);
        }
        else
        {
            // Mark that the track is not in a detector
            hit.detector = {};
        }
    }
    else
    {
        // Ensure killed, inactive, and errored tracks don't contribute to hits
        hit.detector = {};
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
