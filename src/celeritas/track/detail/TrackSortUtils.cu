//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.cu
//---------------------------------------------------------------------------//
#include "TrackSortUtils.hh"

#include <random>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceVector.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//

template<class T>
using StateItems
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::device>;

template<class T>
using ThreadItems
    = Collection<T, Ownership::reference, MemSpace::device, ThreadId>;

using TrackSlots = ThreadItems<TrackSlotId::size_type>;

//---------------------------------------------------------------------------//
/*!
 * Partition track_slots based on predicate.
 */
template<class F>
void partition_impl(TrackSlots const& track_slots, F&& func, StreamId stream_id)
{
    auto start = device_pointer_cast(track_slots.data());
    thrust::partition(thrust_execute_on(stream_id),
                      start,
                      start + track_slots.size(),
                      std::forward<F>(func));
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Reorder OpaqueId's based on track_slots so that track_slots[tid] correspond
 * to ids[tid] instead of ids[tacks_slots[tid]].
 */
template<class Id>
__global__ void
reorder_ids_kernel(ObserverPtr<TrackSlotId::size_type const> track_slots,
                   ObserverPtr<Id const> ids,
                   ObserverPtr<typename Id::size_type> ids_out,
                   size_type size)
{
    if (ThreadId tid = celeritas::KernelParamCalculator::thread_id();
        tid < size)
    {
        ids_out.get()[tid.get()]
            = ids.get()[track_slots.get()[tid.get()]].unchecked_get();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sort track slots using ids as keys.
 */
template<class Id, class IdT = typename Id::size_type>
void sort_impl(TrackSlots const& track_slots,
               ObserverPtr<Id const> ids,
               StreamId stream_id)
{
    DeviceVector<IdT> reordered_ids(track_slots.size(), stream_id);
    CELER_LAUNCH_KERNEL_TEMPLATE_1(reorder_ids,
                                   Id,
                                   track_slots.size(),
                                   celeritas::device().stream(stream_id).get(),
                                   track_slots.data(),
                                   ids,
                                   make_observer(reordered_ids.data()),
                                   track_slots.size());
    thrust::sort_by_key(thrust_execute_on(stream_id),
                        reordered_ids.data(),
                        reordered_ids.data() + reordered_ids.size(),
                        device_pointer_cast(track_slots.data()));
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate thread boundaries based on action ID.
 * \pre actions are sorted
 */
__global__ void
tracks_per_action_kernel(ObserverPtr<ActionId const> actions,
                         ObserverPtr<TrackSlotId::size_type const> track_slots,
                         Span<ThreadId> offsets,
                         size_type size)
{
    ThreadId tid = celeritas::KernelParamCalculator::thread_id();
    ActionAccessor get_action{actions, track_slots};

    if ((tid < size) && tid != ThreadId{0})
    {
        ActionId current_action = get_action(tid);
        ActionId previous_action = get_action(tid - 1);
        if (current_action && current_action != previous_action)
        {
            offsets[current_action.unchecked_get()] = tid;
        }
    }
    // needed if the first action range has only one element
    if (tid == ThreadId{0})
    {
        if (ActionId first = get_action(tid))
        {
            offsets[first.unchecked_get()] = tid;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Initialize default threads to track_slots mapping, track_slots[i] = i.
 *
 * TODO: move to global/detail
 */
template<>
void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type> track_slots,
                                        StreamId stream_id)
{
    thrust::sequence(
        thrust_execute_on(stream_id),
        thrust::device_pointer_cast(track_slots.data()),
        thrust::device_pointer_cast(track_slots.data() + track_slots.size()),
        0);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Shuffle track slots.
 *
 * TODO: move to global/detail
 */
template<>
void shuffle_track_slots<MemSpace::device>(
    Span<TrackSlotId::size_type> track_slots, StreamId stream_id)
{
    using result_type = thrust::default_random_engine::result_type;
    thrust::default_random_engine g{
        static_cast<result_type>(track_slots.size())};
    auto start = thrust::device_pointer_cast(track_slots.data());
    thrust::shuffle(
        thrust_execute_on(stream_id), start, start + track_slots.size(), g);
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Sort or partition tracks.
 */
void sort_tracks(DeviceRef<CoreStateData> const& states, TrackOrder order)
{
    switch (order)
    {
        case TrackOrder::partition_status:
            return partition_impl(states.track_slots,
                                  AlivePredicate{states.sim.status.data()},
                                  states.stream_id);
        case TrackOrder::sort_along_step_action:
        case TrackOrder::sort_step_limit_action:
            return sort_impl(states.track_slots,
                             get_action_ptr(states, order),
                             states.stream_id);
        case TrackOrder::sort_particle_type: {
            using Id =
                typename decltype(states.particles.particle_id)::value_type;
            return sort_impl<Id>(states.track_slots,
                                 states.particles.particle_id.data(),
                                 states.stream_id);
        }
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
    DeviceRef<CoreStateData> const& states,
    Span<ThreadId> offsets,
    Collection<ThreadId, Ownership::value, MemSpace::mapped, ActionId>& out,
    TrackOrder order)
{
    CELER_ASSERT(order == TrackOrder::sort_along_step_action
                 || order == TrackOrder::sort_step_limit_action);

    auto start = device_pointer_cast(make_observer(offsets.data()));
    thrust::fill(thrust_execute_on(states.stream_id),
                 start,
                 start + offsets.size(),
                 ThreadId{});
    CELER_DEVICE_CHECK_ERROR();
    auto* stream = celeritas::device().stream(states.stream_id).get();
    CELER_LAUNCH_KERNEL(tracks_per_action,
                        states.size(),
                        stream,
                        get_action_ptr(states, order),
                        states.track_slots.data(),
                        offsets,
                        states.size());

    Span<ThreadId> sout = out[AllItems<ThreadId, MemSpace::mapped>{}];
    Copier<ThreadId, MemSpace::host> copy_to_host{sout, states.stream_id};
    copy_to_host(MemSpace::device, offsets);

    // Copies must be complete before backfilling
    CELER_DEVICE_CALL_PREFIX(StreamSynchronize(stream));
    backfill_action_count(sout, states.size());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
