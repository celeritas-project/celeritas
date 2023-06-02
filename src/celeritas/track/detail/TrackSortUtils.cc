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

// PRE: action_accessor is sorted, i.e. i <= j ==> action_accessor(i) <=
// action_accessor(j)
template<class F>
void count_tracks_per_action_impl(Span<ThreadId> offsets,
                                  size_type size,
                                  F&& action_accessor)
{
    std::fill(offsets.begin(), offsets.end(), ThreadId{});

    // won't initialize 1st action range
#pragma omp parallel for
    for (size_type i = 1; i < size; ++i)
    {
        ActionId current_action = action_accessor(ThreadId{i});
        if (!current_action)
            continue;

        if (current_action != action_accessor(ThreadId{i - 1}))
        {
            offsets[current_action.get()] = ThreadId{i};
        }
    }

    // needed if the first action range has only one element
    if (ActionId first = action_accessor(ThreadId{0}))
    {
        offsets[first.get()] = ThreadId{0};
    }
    backfill_action_count(offsets, size);
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
/*!
 * Count tracks associated to each action that was used to sort them, specified
 * by order. Result is written in the output parameter offsets which sould be
 * of size num_actions + 1.
 */
template<>
void count_tracks_per_action<MemSpace::host>(
    HostRef<CoreStateData> const& states,
    Span<ThreadId> offsets,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>&,
    TrackOrder order)
{
    switch (order)
    {
        case TrackOrder::sort_along_step_action:
            return count_tracks_per_action_impl(
                offsets,
                states.size(),
                along_step_action_accessor{states.sim.along_step_action.data(),
                                           states.track_slots.data()});
        case TrackOrder::sort_step_limit_action:
            return count_tracks_per_action_impl(
                offsets,
                states.size(),
                step_limit_action_accessor{states.sim.step_limit.data(),
                                           states.track_slots.data()});
        default:
            return;
    }
}

void backfill_action_count(Span<ThreadId> offsets, size_type size)
{
    // offsets.size() == num_actions + 1, have the last offsets be the # of
    // tracks for backfilling correct values in case the last actions are not
    // present
    offsets.back() = ThreadId{size};

    // in case some actions were not found, have them "start" at the next
    // action offset.
    for (auto thread_id = offsets.end() - 2; thread_id >= offsets.begin();
         --thread_id)
    {
        if (*thread_id == ThreadId{})
        {
            *thread_id = *(thread_id + 1);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
