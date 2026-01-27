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
template<>
copy_hits(DetectorHitOutput* output,
          DetectorStateData<Ownership::reference, MemSpace::device> const& state)
{
    CELER_EXPECT(output);

    // Trivially copy all track hits from device

    size_type num_tracks = state.all_track_hits.size();

    output->hits.resize(num_tracks);
    Copier<DetectorHit, MemSpace::host> copy{{output->hits.data(), num_tracks},
                                             state.stream_id};
    copy(MemSpace::device, {state.all_track_hits.data().get(), num_tracks});

    CELER_DEVICE_API_CALL(
        StreamSynchronize(celeritas::device().stream(state.stream_id).get()));

    CELER_ENSURE(output->hits.size() == num_tracks);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
