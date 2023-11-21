//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

// Initialize default threads to track_slots mapping, track_slots[i] = i
// TODO: move to global/detail and overload using ObserverPtr
template<MemSpace M,
         typename Size,
         typename = std::enable_if_t<std::is_unsigned_v<Size>>>
void fill_track_slots(Span<Size> track_slots, StreamId);

template<>
void fill_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots,
                                      StreamId);
template<>
void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type> track_slots,
                                        StreamId);

//---------------------------------------------------------------------------//
// Shuffle tracks
// TODO: move to global/detail and overload using ObserverPtr
template<MemSpace M,
         typename Size,
         typename = std::enable_if_t<std::is_unsigned_v<Size>>>
void shuffle_track_slots(Span<Size> track_slots, StreamId);

template<>
void shuffle_track_slots<MemSpace::host>(
    Span<TrackSlotId::size_type> track_slots, StreamId);
template<>
void shuffle_track_slots<MemSpace::device>(
    Span<TrackSlotId::size_type> track_slots, StreamId);

//---------------------------------------------------------------------------//
// Sort or partition tracks
void sort_tracks(HostRef<CoreStateData> const&, TrackOrder);
void sort_tracks(DeviceRef<CoreStateData> const&, TrackOrder);

//---------------------------------------------------------------------------//
// Count tracks associated to each action

void count_tracks_per_action(
    HostRef<CoreStateData> const&,
    Span<ThreadId>,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>&,
    TrackOrder);

void count_tracks_per_action(
    DeviceRef<CoreStateData> const&,
    Span<ThreadId>,
    Collection<ThreadId, Ownership::value, MemSpace::mapped, ActionId>&,
    TrackOrder);

void backfill_action_count(Span<ThreadId>, size_type);

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
struct alive_predicate
{
    ObserverPtr<TrackStatus const> status_;

    CELER_FUNCTION bool operator()(size_type track_slot) const
    {
        return status_.get()[track_slot] == TrackStatus::alive;
    }
};

struct action_comparator
{
    ObserverPtr<ActionId const> action_;

    CELER_FUNCTION bool operator()(size_type a, size_type b) const
    {
        return action_.get()[a] < action_.get()[b];
    }
};

struct ActionAccessor
{
    ObserverPtr<ActionId const> action_;
    ObserverPtr<TrackSlotId::size_type const> track_slots_;

    CELER_FUNCTION ActionId operator()(ThreadId tid) const
    {
        return action_.get()[track_slots_.get()[tid.get()]];
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<>
inline void
fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type>, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<>
inline void
shuffle_track_slots<MemSpace::device>(Span<TrackSlotId::size_type>, StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void sort_tracks(DeviceRef<CoreStateData> const&, TrackOrder)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void count_tracks_per_action(
    DeviceRef<CoreStateData> const&,
    Span<ThreadId>,
    Collection<ThreadId, Ownership::value, MemSpace::mapped, ActionId>&,
    TrackOrder)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
