//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detector/DetectorData.cu
//---------------------------------------------------------------------------//
#include "DetectorData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Copy hits from device to pinned memory.
 *
 * All hits from all tracks are copied. These may include invalid hits where a
 * track is not in a detector, which is indicated by an invalid detector ID.
 * The user of the output is therefore responsible for parsing the pinned
 * memory for only valid hits.
 */
template<>
copy_hits(DetectorHitOutput* output,
          DetectorStateData<Ownership::reference, MemSpace::device> const& state,
          StreamId stream_id)
{
    CELER_EXPECT(output);
    CELER_EXPECT(stream_id);

    size_type num_tracks = state.all_track_hits.size();

    // Copy all track hits from device
    output->hits.resize(num_tracks);
    Copier<DetectorHit, MemSpace::host> copy{{output->hits.data(), num_tracks},
                                             stream_id};
    copy(MemSpace::device, {state.all_track_hits.data().get(), num_tracks});

    // Synchronize to ensure all data is transferred before continuing
    CELER_DEVICE_API_CALL(
        StreamSynchronize(celeritas::device().stream(stream_id).get()));

    CELER_ENSURE(output->hits.size() == num_tracks);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
