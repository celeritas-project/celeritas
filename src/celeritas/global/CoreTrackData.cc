//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/CoreTrackData.cc
//---------------------------------------------------------------------------//
#include "CoreTrackData.hh"

#include "celeritas/track/detail/TrackSortUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 */
template<MemSpace M>
void resize(CoreStateData<Ownership::value, M>* state,
            HostCRef<CoreParamsData> const& params,
            StreamId stream_id,
            size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(stream_id);
    CELER_EXPECT(size > 0);
    CELER_VALIDATE(stream_id < params.scalars.max_streams,
                   << "multitasking stream_id=" << stream_id.unchecked_get()
                   << " exceeds max_streams=" << params.scalars.max_streams);
#if CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_GEANT4
    resize(&state->geometry, params.geometry, size);
#else
    // Geant4 state is stream-local
    resize(&state->geometry, params.geometry, stream_id, size);
#endif
    resize(&state->materials, params.materials, size);
    resize(&state->particles, params.particles, size);
    resize(&state->physics, params.physics, size);
    resize(&state->rng, params.rng, stream_id, size);
    resize(&state->sim, size);
    resize(&state->init, params.init, size);
    resize(&state->track_slots, size);
    state->stream_id = stream_id;

    Span track_slots{state->track_slots[AllItems<TrackSlotId::size_type, M>{}]};
    detail::fill_track_slots<M>(track_slots, stream_id);
    if (params.init.track_order == TrackOrder::shuffled)
    {
        detail::shuffle_track_slots<M>(track_slots, stream_id);
    }

    CELER_ENSURE(state);
}

//---------------------------------------------------------------------------//
template void
resize<MemSpace::host>(CoreStateData<Ownership::value, MemSpace::host>*,
                       HostCRef<CoreParamsData> const&,
                       StreamId,
                       size_type);

template void
resize<MemSpace::device>(CoreStateData<Ownership::value, MemSpace::device>*,
                         HostCRef<CoreParamsData> const&,
                         StreamId,
                         size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
