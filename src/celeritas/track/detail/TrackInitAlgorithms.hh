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
#include "corecel/data/ObserverPtr.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreParams.hh"

#include "../TrackInitData.hh"
#include "../Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Flag the initializers that will become neutral tracks in this step.
 *
 * The argument is an index into the initializers used to create new tracks
 * in this step, which are the last \c min(num_vacancies,num_initializers)
 * elements of the initializer storage. Indices past that range are flagged as
 * zero so that the flags can be scanned over a fixed size (the number of track
 * slots) without the host knowing the number of new tracks.
 */
struct IsNeutralNewTrack
{
    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;

    ParamsPtr params;
    TrackInitializer const* initializers{nullptr};
    CoreStateCounters const* counters{nullptr};

    CELER_FUNCTION size_type operator()(size_type i) const
    {
        CELER_EXPECT(initializers && counters);
        size_type num_new
            = min(counters->num_vacancies, counters->num_initializers);
        if (i >= num_new)
        {
            return 0;
        }
        TrackInitializer const& init
            = initializers[counters->num_initializers - num_new + i];
        return IsNeutral{params}(init) ? 1 : 0;
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
ObserverPtr<size_type, MemSpace::host> exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::host> const&,
    StreamId);
ObserverPtr<size_type, MemSpace::device> exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&,
    StreamId);

//---------------------------------------------------------------------------//
// Count the neutral tracks that will be initialized in this step
void scan_neutral_initializers(
    CoreParams const&,
    TrackInitStateData<Ownership::reference, MemSpace::host> const&,
    StreamId);
void scan_neutral_initializers(
    CoreParams const&,
    TrackInitStateData<Ownership::reference, MemSpace::device> const&,
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

inline ObserverPtr<size_type, MemSpace::device> exclusive_scan_counts(
    StateCollection<size_type, Ownership::reference, MemSpace::device> const&,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void scan_neutral_initializers(
    CoreParams const&,
    TrackInitStateData<Ownership::reference, MemSpace::device> const&,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
