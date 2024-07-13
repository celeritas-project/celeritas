//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackInitAlgorithms.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/ParticleView.hh"
#include "celeritas/track/CoreStateCounters.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Predicate for sorting charged from neutral tracks
struct IsNeutral
{
    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;

    ParamsPtr params;

    CELER_FUNCTION bool operator()(TrackInitializer ti) const
    {
        return ParticleView(params->particles, ti.particle.particle_id).charge()
               == zero_quantity();
    }
};

//---------------------------------------------------------------------------//
// Remove all elements in the vacancy vector that were flagged as alive
size_type remove_if_alive(
    StateCollection<TrackSlotId, Ownership::reference, MemSpace::host> const&,
    StreamId);
size_type remove_if_alive(
    StateCollection<TrackSlotId, Ownership::reference, MemSpace::device> const&,
    StreamId);

//---------------------------------------------------------------------------//
// Calculate the exclusive prefix sum of the number of surviving secondaries
size_type exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::host> const&,
    StreamId);
size_type exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&,
    StreamId);

//---------------------------------------------------------------------------//
// Sort the tracks that will be initialized in this step by charged/neutral
size_type partition_initializers(
    CoreParams const&,
    Collection<TrackInitializer, Ownership::reference, MemSpace::host> const&,
    CoreStateCounters const&,
    size_type,
    StreamId);
size_type partition_initializers(
    CoreParams const&,
    Collection<TrackInitializer, Ownership::reference, MemSpace::device> const&,
    CoreStateCounters const&,
    size_type,
    StreamId);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline size_type remove_if_alive(
    StateCollection<TrackSlotId, Ownership::reference, MemSpace::device> const&,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline size_type exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline size_type partition_initializers(
    CoreParams const&,
    Collection<TrackInitializer, Ownership::reference, MemSpace::device> const&,
    CoreStateCounters const&,
    size_type,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
