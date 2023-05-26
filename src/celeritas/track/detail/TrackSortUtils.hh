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
void fill_track_slots(Span<Size> track_slots);

template<>
void fill_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots);
template<>
void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type> track_slots);

//---------------------------------------------------------------------------//
// Shuffle tracks
// TODO: move to global/detail and overload using ObserverPtr
template<MemSpace M,
         typename Size,
         typename = std::enable_if_t<std::is_unsigned_v<Size>>>
void shuffle_track_slots(Span<Size> track_slots);

template<>
void shuffle_track_slots<MemSpace::host>(
    Span<TrackSlotId::size_type> track_slots);
template<>
void shuffle_track_slots<MemSpace::device>(
    Span<TrackSlotId::size_type> track_slots);

//---------------------------------------------------------------------------//
// Sort or partition tracks
void sort_tracks(HostRef<CoreStateData> const&, TrackOrder);
void sort_tracks(DeviceRef<CoreStateData> const&, TrackOrder);

//---------------------------------------------------------------------------//
// Count tracks associated to each action
template<MemSpace M>
void count_tracks_per_action(
    CoreStateData<Ownership::reference, M> const&,
    Span<ThreadId>,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>&,
    TrackOrder);

template<>
void count_tracks_per_action<MemSpace::host>(
    HostRef<CoreStateData> const&,
    Span<ThreadId>,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>&,
    TrackOrder);
template<>
void count_tracks_per_action<MemSpace::device>(
    DeviceRef<CoreStateData> const&,
    Span<ThreadId>,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>&,
    TrackOrder);

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//
struct alive_predicate
{
    ObserverPtr<TrackStatus const> status_;

    CELER_FUNCTION bool operator()(unsigned int track_slot) const
    {
        return status_.get()[track_slot] == TrackStatus::alive;
    }
};

struct step_limit_comparator
{
    ObserverPtr<StepLimit const> step_limit_;

    CELER_FUNCTION bool operator()(unsigned int a, unsigned int b) const
    {
        return step_limit_.get()[a].action < step_limit_.get()[b].action;
    }
};

struct along_action_comparator
{
    ObserverPtr<ActionId const> action_;

    CELER_FUNCTION bool operator()(unsigned int a, unsigned int b) const
    {
        return action_.get()[a] < action_.get()[b];
    }
};

struct step_limit_action_accessor
{
    ObserverPtr<StepLimit const> step_limit_;
    ObserverPtr<TrackSlotId::size_type const> track_slots_;

    CELER_FUNCTION ActionId operator()(ThreadId tid) const
    {
        return step_limit_.get()[track_slots_.get()[tid.get()]].action;
    }
};

struct along_step_action_accessor
{
    ObserverPtr<ActionId const> along_step_;
    ObserverPtr<TrackSlotId::size_type const> track_slots_;

    CELER_FUNCTION ActionId operator()(ThreadId tid) const
    {
        return along_step_.get()[track_slots_.get()[tid.get()]];
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<>
inline void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<>
inline void shuffle_track_slots<MemSpace::device>(Span<TrackSlotId::size_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void sort_tracks(DeviceRef<CoreStateData> const&, TrackOrder)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<>
inline void count_tracks_per_action<MemSpace::device>(
    DeviceRef<CoreStateData> const&,
    Span<ThreadId>,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>&,
    TrackOrder)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
