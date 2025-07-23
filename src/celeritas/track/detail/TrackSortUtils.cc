//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.cc
//---------------------------------------------------------------------------//
#include "TrackSortUtils.hh"

#include <algorithm>
#include <iterator>
#include <numeric>

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

//---------------------------------------------------------------------------//
//! Partition track slots based on a unary predicate
template<class F>
void partition_impl(TrackSlots const& track_slots, F&& func)
{
    auto* start = track_slots.data().get();
    std::partition(start, start + track_slots.size(), std::forward<F>(func));
}

//---------------------------------------------------------------------------//
//! Sort track slots based on a binary predicate
template<class F>
void sort_impl(TrackSlots const& track_slots, F&& func)
{
    auto* start = track_slots.data().get();
    std::sort(start, start + track_slots.size(), std::forward<F>(func));
}

//---------------------------------------------------------------------------//
//! Compare indices by indirection from a track slot index
template<class Id>
struct IdLess
{
    ObserverPtr<Id const> ids_;

    bool operator()(size_type a, size_type b) const
    {
        return ids_.get()[a] < ids_.get()[b];
    }
};

template<class Id>
IdLess(ObserverPtr<Id>) -> IdLess<Id>;

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Sort or partition tracks.
 */
void sort_tracks(HostRef<CoreStateData> const& states, TrackOrder order)
{
    switch (order)
    {
        case TrackOrder::reindex_status:
            return partition_impl(states.track_slots,
                                  IsNotInactive{states.sim.status.data()});
        case TrackOrder::reindex_along_step_action:
        case TrackOrder::reindex_step_limit_action:
            return sort_impl(states.track_slots,
                             IdLess{get_action_ptr(states, order)});
        case TrackOrder::reindex_particle_type:
            return sort_impl(states.track_slots,
                             IdLess{states.particles.particle_id.data()});
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Count tracks associated to each action that was used to sort them, specified
 * by order. Result is written in the output parameter offsets which should be
 * of size num_actions + 1.
 */
void count_tracks_per_action(
    HostRef<CoreStateData> const& states,
    Span<ThreadId> offsets,
    Collection<ThreadId, Ownership::value, MemSpace::host, ActionId>&,
    TrackOrder order)
{
    CELER_ASSERT(order == TrackOrder::reindex_along_step_action
                 || order == TrackOrder::reindex_step_limit_action);

    ActionAccessor get_action{get_action_ptr(states, order),
                              states.track_slots.data()};

    std::fill(offsets.begin(), offsets.end(), ThreadId{});
    auto const size = states.size();
    // if get_action(0) != get_action(1), get_action(0) never gets initialized
#if CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
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
