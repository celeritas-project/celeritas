//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/DetectorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Range.hh"
#include "celeritas/Types.hh"
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
 * or killed, or is not in a detector region, an invalid hit is set in the
 * corresponding buffer track slot.
 *
 * This action runs at \c user_post, after surface interactions (which absorb
 * photons and set \c TrackStatus::killed ). Both \c alive and \c killed tracks
 * are scored, analogous to how the EM \c StepGatherExecutor handles killed
 * tracks. Inactive and errored tracks are skipped.
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
    // Clear the hit if inactive, errored, or not detected
    hit.detector = {};

    auto sim = track.sim();

    auto const status = sim.status();
    if (status == TrackStatus::inactive || status == TrackStatus::errored)
    {
        // Skip empty slots and errored tracks
        hit.detector = {};
        return;
    }

    auto const detectors = track.detectors();
    auto geometry = track.geometry();
    auto const volume_id = geometry.volume_id();
    auto const detector_id = detectors.detector_id(volume_id);

    if (detector_id)
    {
        // Score a valid hit for alive or killed tracks in a detector volume
        hit = DetectorHit{detector_id,
                          track.particle().energy(),
                          sim.time(),
                          geometry.pos(),
                          geometry.dir(),
                          geometry.volume_instance_id()};

        // Store full volume hierarchy if buffer is allocated
        auto const num_levels = detector_state_.num_volume_levels;
        if (num_levels > 0)
        {
            auto const tid = track.track_slot_id();
            auto all_ids
                = detector_state_
                      .volume_instance_ids[AllItems<VolumeInstanceId>{}];
            auto dst = all_ids.subspan(tid.unchecked_get() * num_levels,
                                       num_levels);
            size_type depth = geometry.volume_level().unchecked_get() + 1;
            CELER_ASSERT(depth <= dst.size());
            geometry.volume_instance_id(dst.first(depth));
            for (auto level : range<size_type>(depth, num_levels))
            {
                dst[level] = {};
            }
        }
    }
    else
    {
        // Track is not in a detector volume
        hit.detector = {};
    }

    // Score a valid hit
    hit.detector = detector_id;
    hit.primary = sim.primary_id();
    hit.energy = track.particle().energy();
    hit.time = sim.time();
    hit.position = geometry.pos();
    hit.volume_instance = geometry.volume_instance_id();

    // Kill the track
    sim.status(TrackStatus::killed);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
