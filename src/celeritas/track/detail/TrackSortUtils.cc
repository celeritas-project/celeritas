//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.cc
//---------------------------------------------------------------------------//
#include "TrackSortUtils.hh"

#include <algorithm>
#include <iterator>
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

// PRE: get_action is sorted, i.e. i <= j ==> get_action(i) <=
// get_action(j)
template<class F>
void count_tracks_per_action_impl(Span<ThreadId> offsets,
                                  size_type size,
                                  F&& get_action)
{
    std::fill(offsets.begin(), offsets.end(), ThreadId{});

    // if get_action(0) != get_action(1), get_action(0) never gets initialized
#pragma omp parallel for
    for (size_type i = 1; i < size; ++i)
    {
        ActionId current_action = get_action(ThreadId{i});
        if (!current_action)
            continue;

        if (current_action != get_action(ThreadId{i - 1}))
        {
            offsets[current_action.unchecked_get()] = ThreadId{i};
        }
    }

    // so make sure get_action(0) is initialized
    if (ActionId first = get_action(ThreadId{0}))
    {
        offsets[first.unchecked_get()] = ThreadId{0};
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
void fill_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots,
                                      StreamId)
{
    std::iota(track_slots.data(), track_slots.data() + track_slots.size(), 0);
}

/*!
 * Shuffle track slots
 */
template<>
void shuffle_track_slots<MemSpace::host>(
    Span<TrackSlotId::size_type> track_slots, StreamId)
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
                action_comparator{states.sim.along_step_action.data()},
                states.stream_id);
        case TrackOrder::sort_step_limit_action:
            return sort_impl(
                states.track_slots,
                action_comparator{states.sim.post_step_action.data()},
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
void count_tracks_per_action(
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
                ActionAccessor{states.sim.along_step_action.data(),
                               states.track_slots.data()});
        case TrackOrder::sort_step_limit_action:
            return count_tracks_per_action_impl(
                offsets,
                states.size(),
                ActionAccessor{states.sim.post_step_action.data(),
                               states.track_slots.data()});
        default:
            return;
    }
}

void backfill_action_count(Span<ThreadId> offsets, size_type num_actions)
{
    CELER_EXPECT(offsets.size() >= 2);
    // offsets.size() == num_actions + 1, have the last offsets be the # of
    // tracks for backfilling correct values in case the last actions are not
    // present
    offsets.back() = ThreadId{num_actions};

    // in case some actions were not found, have them "start" at the next
    // action offset.
    for (auto thread_id = std::reverse_iterator(offsets.end() - 1);
         thread_id != std::reverse_iterator(offsets.begin());
         ++thread_id)
    {
        if (!*thread_id)
        {
            *thread_id = *(thread_id - 1);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
