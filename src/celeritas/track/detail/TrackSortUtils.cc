//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
    auto seed = static_cast<unsigned int>(track_slots.size());
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
                                  AlivePredicate{states.sim.status.data()});
        case TrackOrder::sort_along_step_action:
        case TrackOrder::sort_step_limit_action:
            return sort_impl(states.track_slots,
                             IdComparator{get_action_ptr(states, order)});
        case TrackOrder::sort_particle_type:
            return sort_impl(states.track_slots,
                             IdComparator{states.particles.particle_id.data()});
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
    CELER_ASSERT(order == TrackOrder::sort_along_step_action
                 || order == TrackOrder::sort_step_limit_action);

    ActionAccessor get_action{get_action_ptr(states, order),
                              states.track_slots.data()};

    std::fill(offsets.begin(), offsets.end(), ThreadId{});
    auto const size = states.size();
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
/*!
 * Fill missing action offsets.
 */
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
