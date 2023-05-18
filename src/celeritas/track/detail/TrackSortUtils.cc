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
void partition_impl(TrackSlots const& track_slots, F&& func)
{
    auto* start = track_slots.data().get();
    std::partition(start, start + track_slots.size(), std::forward<F>(func));
}

//---------------------------------------------------------------------------//

template<class F>
void sort_impl(TrackSlots const& track_slots, F&& func)
{
    auto* start = track_slots.data().get();
    std::sort(start, start + track_slots.size(), std::forward<F>(func));
}

template<class F>
void tracks_per_action(HostRef<CoreStateData> const& states,
                       F&& action_accessor)
{
    Span<ThreadId> offsets = states.thread_offsets[AllItems<ThreadId>{}];
    std::fill(offsets.begin(), offsets.end(), ThreadId{});
// won't initialize 1st action range
#pragma omp parallel for
    for (size_type i = 1; i < states.size(); ++i)
    {
        ActionId current_action = action_accessor(ThreadId{i});
        if (!current_action)
            continue;

        ActionId previous_action = action_accessor(ThreadId{i - 1});
        if (current_action != previous_action)
        {
            offsets[current_action.get()] = ThreadId{i};
        }
    }

    // Initialize the offset for the 1st action range if valid, always starts
    // at 0
    ActionId first = action_accessor(ThreadId{0});
    if (first)
    {
        offsets[first.get()] = ThreadId{0};
    }
    // offsets.size() == num_actions + 1, have the last offsets be the # of
    // threads for backfilling in case the last actions are not present
    offsets.back() = ThreadId{states.size()};

    // in case some actions where not found, have them "start" at the next
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
                                  alive_predicate{states.sim.status.data()});
        case TrackOrder::sort_along_step_action:
            sort_impl(
                states.track_slots,
                along_action_comparator{states.sim.along_step_action.data()});
            tracks_per_action(
                states,
                [&along_step_action = states.sim.along_step_action,
                 &track_slots = states.track_slots](ThreadId tid) -> ActionId {
                    return along_step_action[TrackSlotId{track_slots[tid]}];
                });
            return;
        case TrackOrder::sort_step_limit_action:
            sort_impl(states.track_slots,
                      step_limit_comparator{states.sim.step_limit.data()});
            tracks_per_action(
                states,
                [&step_limit = states.sim.step_limit,
                 &track_slots = states.track_slots](ThreadId tid) -> ActionId {
                    return step_limit[TrackSlotId{track_slots[tid]}].action;
                });
            return;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
