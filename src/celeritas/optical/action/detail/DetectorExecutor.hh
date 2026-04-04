//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/DetectorExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

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
 * or is not in a detector region, an invalid hit is set in the corresponding
 * buffer track slot.
 *
 * Detection can occur after a surface crossing \em or (for coupled EM
 * problems) if an optical photon is emitted inside a detector region.
 *
 * When a track generates a valid hit, it is killed (absorbed by the detector).
 *
 * \note This is called by \em all track slots, which is necessary to clear the
 * outgoing detector hits. We could break this into a "reset" kernel that
 * applies to all slots, and a "fill" kernel that applies only to valid slots.
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

    if (!is_track_valid(sim.status()))
    {
        // Inactive or errored
        return;
    }
    if (track.surface_physics().is_crossing_boundary())
    {
        // Boundary crossing not yet completed, so not yet detected!
        return;
    }

    auto geometry = track.geometry();
    if (geometry.is_outside())
    {
        // Killed by leaving geometry; no detection
        return;
    }

    auto const detectors = track.detectors();
    auto const volume_id = geometry.volume_id();
    auto const detector_id = detectors.detector_id(volume_id);

    if (!detector_id)
    {
        // Not in a detector
        return;
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
