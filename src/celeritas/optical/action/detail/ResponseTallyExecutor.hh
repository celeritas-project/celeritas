//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/ResponseTallyExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

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
 * Accumulate detector hits into the response tally.
 *
 * A photon is scored if it was killed by the detector action this step: it
 * has \c killed status, is not in the middle of a boundary crossing, is
 * inside the geometry, and is in a volume mapped to a detector. These are the
 * same criteria used by \c DetectorExecutor to register a hit, so this must
 * execute \em after the detector action. The compressed bin ID is saved in
 * the per-track state, and the stream's bin count is atomically incremented.
 */
struct ResponseTallyExecutor
{
    NativeCRef<ResponseTallyParamsData> const params;
    NativeRef<ResponseTallyStateData> const state;

    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

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

    auto& bin = state.bin_ids[track.track_slot_id()];
    bin = ResponseTallyView::invalid_bin();

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

    ResponseTallyView view(params);
    bin = view.bin_id(detector_id.unchecked_get(), geo.pos(), sim.time());
    if (bin != ResponseTallyView::invalid_bin())
    {
        atomic_add(&state.counts[ItemId<size_type>(bin)], 1_sz);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
