//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Atomics.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/ParticleView.hh"

#include "TrackInitData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Predicate for sorting charged from neutral tracks
struct IsNeutral
{
    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;

    ParamsPtr params;

    CELER_FUNCTION bool operator()(TrackInitializer const& ti) const
    {
        return ParticleView(params->particles, ti.particle.particle_id).charge()
               == zero_quantity();
    }
};

//---------------------------------------------------------------------------//
//! Get an initializer index where thread 0 has the last valid element
CELER_FORCEINLINE_FUNCTION size_type index_before(size_type size, ThreadId tid)
{
    CELER_EXPECT(tid.get() + 1 <= size);
    return size - tid.unchecked_get() - 1;
}

//---------------------------------------------------------------------------//
/*!
 * Create a unique track ID for the given event.
 *
 * \todo This is nondeterministic; we need to calculate the track ID in a
 * reproducible way.
 */
inline CELER_FUNCTION TrackId
make_track_id(NativeCRef<TrackInitParamsData> const&,
              NativeRef<TrackInitStateData>& state,
              EventId event)
{
    CELER_EXPECT(event < state.track_counters.size());
    auto result
        = atomic_add(&state.track_counters[event], TrackId::size_type{1});
    return TrackId{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
