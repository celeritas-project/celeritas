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
    NativeRef<DetectorStateData> data;

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
    auto const detectors = track.detectors();

    auto const volume_id = track.geometry().volume_id();
    auto const detector_id = detectors.detector_id(volume_id);

    DetectorHit& hit = data.all_track_hits[track.track_slot_id()];
    hit.detector = detector_id;

    if (detector_id)
    {
        // Populate hit data
        hit.energy = track.particle().energy();
        hit.time = track.sim().time();
        hit.position = track.geometry().pos();
        hit.volume_instance = track.geometry().volume_instance_id();

        // Kill the track
    }
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
