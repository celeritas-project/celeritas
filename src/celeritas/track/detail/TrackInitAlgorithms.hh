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

#include "../CoreStateCounters.hh"
#include "../TrackInitData.hh"
#include "../Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Predicate for separating charged from neutral tracks with a stencil
struct IsNeutralStencil
{
    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;

    ParamsPtr params;
    TrackInitializer const* initializers;

    CELER_FUNCTION bool operator()(size_type i) const
    {
        CELER_EXPECT(initializers);
        return IsNeutral{params}(initializers[i]);
    }
};

//---------------------------------------------------------------------------//
// Remove all elements in the vacancy vector that were flagged as alive
void remove_if_alive(
    TrackInitStateData<Ownership::reference, MemSpace::host> const&, StreamId);
void remove_if_alive(
    TrackInitStateData<Ownership::reference, MemSpace::device> const&,
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
    size_type,
    StreamId);
void partition_initializers(
    CoreParams const&,
    TrackInitStateData<Ownership::reference, MemSpace::device> const&,
    size_type,
    StreamId);

//---------------------------------------------------------------------------//
// DEVICE-DISABLED IMPLEMENTATION
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline void remove_if_alive(
    TrackInitStateData<Ownership::reference, MemSpace::device> const&, StreamId)
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
    size_type,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
