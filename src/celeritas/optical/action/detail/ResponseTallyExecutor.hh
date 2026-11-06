//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/ResponseTallyExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/ResponseTallyData.hh"
#include "celeritas/optical/ResponseTallyView.hh"
#include "celeritas/optical/SimTrackView.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Save the spatial bin of newly initialized tracks.
 *
 * This must execute after track initialization and before the first
 * along-step, when a new track is in the \c initializing state and still at
 * its starting position. For a secondary (e.g., wavelength shifted) photon,
 * this is the point where the secondary is emitted. The detector and time
 * parts of the bin are added by \c ResponseTallyExecutor when the track hits
 * a detector.
 */
struct ResponseTallyStartExecutor
{
    NativeCRef<ResponseTallyParamsData> const params;
    NativeRef<ResponseTallyStateData> const state;

    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Accumulate detector hits into the response tally.
 *
 * A photon is scored if it was killed by the detector action this step: it
 * has \c killed status, is not in the middle of a boundary crossing, is
 * inside the geometry, and is in a volume mapped to a detector. These are the
 * same criteria used by \c DetectorExecutor to register a hit, so this must
 * execute \em after the detector action. The detector and time of the hit
 * complete the starting bin saved by \c ResponseTallyStartExecutor, and the
 * stream's bin count is atomically incremented.
 *
 * The saved bin is cleared once a track has been scored, so a track can never
 * be counted more than once.
 */
struct ResponseTallyExecutor
{
    NativeCRef<ResponseTallyParamsData> const params;
    NativeRef<ResponseTallyStateData> const state;

    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Save the spatial bin of a newly initialized track.
 */
CELER_FUNCTION void ResponseTallyStartExecutor::operator()(
    CoreTrackView const& track)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    if (track.sim().status() != TrackStatus::initializing)
    {
        return;
    }

    ResponseTallyView view(params);
    state.bin_ids[track.track_slot_id()]
        = view.start_bin(track.geometry().pos());
}

//---------------------------------------------------------------------------//
/*!
 * Score a hit in the tally bins.
 */
CELER_FUNCTION void ResponseTallyExecutor::operator()(
    CoreTrackView const& track)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    using namespace celeritas::literals;

    auto sim = track.sim();
    if (sim.status() != TrackStatus::killed)
    {
        return;
    }
    if (track.surface_physics().is_crossing_boundary())
    {
        return;
    }

    auto geo = track.geometry();
    if (geo.is_outside())
    {
        return;
    }

    auto detector_id = track.detectors().detector_id(geo.volume_id());
    if (!detector_id)
    {
        return;
    }

    auto& start_bin = state.bin_ids[track.track_slot_id()];
    ResponseTallyView view(params);
    auto bin = view.end_bin(start_bin, detector_id, sim.time());

    // Always clear so the track cannot be scored twice
    start_bin = nullid;

    if (bin)
    {
        atomic_add(&state.counts[bin], 1_sz);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
