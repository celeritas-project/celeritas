//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
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
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"

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

template<class F>
void partition_impl(TrackSlots const& track_slots, F&& func)
{
    auto start = device_pointer_cast(track_slots.data());
    thrust::partition(thrust::device,
                      start,
                      start + track_slots.size(),
                      std::forward<F>(func));
    CELER_DEVICE_CHECK_ERROR();
}

//---------------------------------------------------------------------------//

template<class F>
void sort_impl(TrackSlots const& track_slots, F&& func)
{
    auto start = device_pointer_cast(track_slots.data());
    thrust::sort(thrust::device,
                 start,
                 start + track_slots.size(),
                 std::forward<F>(func));
    CELER_DEVICE_CHECK_ERROR();
}

template<class F>
CELER_FUNCTION void
tracks_per_action_impl(DeviceRef<CoreStateData> const& states,
                       F&& action_accessor)
{
    ThreadId tid = celeritas::KernelParamCalculator::thread_id();

    if ((tid < states.size()) && tid.get() != 0)
    {
        Span<ThreadId> offsets = states.thread_offsets[AllItems<ThreadId>{}];
        ActionId current_action = action_accessor(tid);
        ActionId previous_action = action_accessor(tid - 1);
        if (current_action && current_action != previous_action)
        {
            offsets[current_action.get()] = tid;
        }
    }
}

// PRE: action_accessor is sorted, i.e. i <= j ==> action_accessor(i) <=
// action_accessor(j)
__global__ void tracks_per_action_kernel(DeviceRef<CoreStateData> const states,
                                         TrackOrder order)
{
    // DISPATCH here since CELER_LAUNCH_KERNEL doesn't work with templated
    // kernels
    switch (order)
    {
        case TrackOrder::sort_along_step_action:
            return tracks_per_action_impl(
                states,
                along_step_action_accessor{states.sim.along_step_action.data(),
                                           states.track_slots.data()});
        case TrackOrder::sort_step_limit_action:
            return tracks_per_action_impl(
                states,
                step_limit_action_accessor{states.sim.step_limit.data(),
                                           states.track_slots.data()});
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

// TODO: On host
__global__ void
tracks_per_action_reduce_kernel(DeviceRef<CoreStateData> const states)
{
    ThreadId tid = celeritas::KernelParamCalculator::thread_id();
    if (tid.get() == 0)
    {
        Span<ThreadId> offsets = states.thread_offsets[AllItems<ThreadId>{}];
        offsets.back() = ThreadId{states.size()};
        for (auto thread_id = offsets.end() - 2; thread_id >= offsets.begin();
             --thread_id)
        {
            if (*thread_id == ThreadId{})
            {
                *thread_id = *(thread_id + 1);
            }
        }
    }
}

void tracks_per_action(DeviceRef<CoreStateData> const& states, TrackOrder order)
{
    auto start = device_pointer_cast(states.thread_offsets.data());
    thrust::fill(start, start + states.thread_offsets.size(), ThreadId{});
    CELER_DEVICE_CHECK_ERROR();

    CELER_LAUNCH_KERNEL(tracks_per_action,
                        celeritas::device().default_block_size(),
                        states.size(),
                        states,
                        order);
    CELER_DEVICE_CHECK_ERROR();

    CELER_LAUNCH_KERNEL(tracks_per_action_reduce,
                        celeritas::device().default_block_size(),
                        states.size(),
                        states);
    CELER_DEVICE_CHECK_ERROR();
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
void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type> track_slots)
{
    thrust::sequence(
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
    Span<TrackSlotId::size_type> track_slots)
{
    using result_type = thrust::default_random_engine::result_type;
    thrust::default_random_engine g{
        static_cast<result_type>(track_slots.size())};
    auto start = thrust::device_pointer_cast(track_slots.data());
    thrust::shuffle(thrust::device, start, start + track_slots.size(), g);
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
                                  alive_predicate{states.sim.status.data()});
        case TrackOrder::sort_along_step_action:
            sort_impl(
                states.track_slots,
                along_action_comparator{states.sim.along_step_action.data()});
            tracks_per_action(states, order);
            return;
        case TrackOrder::sort_step_limit_action:
            sort_impl(states.track_slots,
                      step_limit_comparator{states.sim.step_limit.data()});
            tracks_per_action(states, order);
            return;
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
