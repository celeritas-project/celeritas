//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.cc
//---------------------------------------------------------------------------//
#include "TrackSortUtils.hh"

#include <algorithm>
#include <numeric>
#include <random>

#include "corecel/data/Collection.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//

template<class T>
using ThreadItems
    = Collection<T, Ownership::reference, MemSpace::host, ThreadId>;

using TrackSlots = ThreadItems<TrackSlotId::size_type>;

template<class F>
void partition_impl(TrackSlots const& track_slots, F&& func, StreamId)
{
    auto* start = track_slots.data().get();
    std::partition(start, start + track_slots.size(), std::forward<F>(func));
}

//---------------------------------------------------------------------------//

template<class F>
void sort_impl(TrackSlots const& track_slots, F&& func, StreamId)
{
    auto* start = track_slots.data().get();
    std::sort(start, start + track_slots.size(), std::forward<F>(func));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Initialize default threads to track_slots mapping, track_slots[i] = i
 */
template<>
void fill_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots)
{
    std::iota(track_slots.data(), track_slots.data() + track_slots.size(), 0);
}

/*!
 * Shuffle track slots
 */
template<>
void shuffle_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots)
{
    unsigned int seed = track_slots.size();
    std::mt19937 g{seed};
    std::shuffle(track_slots.begin(), track_slots.end(), g);
}

//---------------------------------------------------------------------------//
/*!
 * Sort or partition tracks.
 */
void sort_tracks(HostRef<CoreStateData> const& states, TrackOrder order)
{
    switch (order)
    {
        case TrackOrder::partition_status:
            return partition_impl(states.track_slots,
                                  alive_predicate{states.sim.status.data()},
                                  states.stream_id);
        case TrackOrder::sort_along_step_action:
            return sort_impl(
                states.track_slots,
                along_action_comparator{states.sim.along_step_action.data()},
                states.stream_id);
        case TrackOrder::sort_step_limit_action:
            return sort_impl(
                states.track_slots,
                step_limit_comparator{states.sim.step_limit.data()},
                states.stream_id);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
