//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "Utils.hh"
#include "../CoreStateCounters.hh"
#include "../TrackInitData.hh"

namespace celeritas
{
namespace detail
{
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
void partition_initializers(
    CoreParams const&,
    TrackInitStateData<Ownership::reference, MemSpace::host> const&,
    CoreStateCounters const&,
    size_type,
    StreamId);
void partition_initializers(
    CoreParams const&,
    TrackInitStateData<Ownership::reference, MemSpace::device> const&,
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

inline void partition_initializers(
    CoreParams const&,
    TrackInitStateData<Ownership::reference, MemSpace::device> const&,
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
