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

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize default threads to track_slots mapping, track_slots[i] = i
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

/*!
 * Shuffle track slots
 */
template<>
void shuffle_track_slots<MemSpace::device>(
    Span<TrackSlotId::size_type> track_slots)
{
    using result_type = thrust::default_random_engine::result_type;
    thrust::default_random_engine g{
        static_cast<result_type>(track_slots.size())};
    thrust::shuffle(
        thrust::device,
        thrust::device_pointer_cast(track_slots.data()),
        thrust::device_pointer_cast(track_slots.data() + track_slots.size()),
        g);
    CELER_DEVICE_CHECK_ERROR();
}

namespace
{
struct alive_predicate
{
    using SpanT = Span<TrackStatus>;
    SpanT status_;

    CELER_FUNCTION explicit alive_predicate(SpanT status) : status_{status} {}
    CELER_FUNCTION bool
    operator()(TrackSlotId::size_type const& track_slot) const
    {
        return status_[track_slot] == TrackStatus::alive;
    }
};

struct step_limit_comparator
{
    using SpanT = Span<StepLimit>;
    SpanT step_limit_;

    CELER_FUNCTION explicit step_limit_comparator(SpanT step_limit)
        : step_limit_{step_limit}
    {
    }
    CELER_FUNCTION bool operator()(TrackSlotId::size_type const& a,
                                   TrackSlotId::size_type const& b) const
    {
        return step_limit_[a].action < step_limit_[b].action;
    }
};

}  // namespace

void partition_tracks_by_status(
    CoreStateData<Ownership::reference, MemSpace::device> const& states)
{
    CELER_EXPECT(states.size() > 0);
    Span track_slots{
        states.track_slots[AllItems<TrackSlotId::size_type, MemSpace::device>{}]};
    thrust::partition(
        thrust::device,
        thrust::device_pointer_cast(track_slots.begin()),
        thrust::device_pointer_cast(track_slots.end()),
        alive_predicate{
            states.sim.status[AllItems<TrackStatus, MemSpace::device>{}]});
    CELER_DEVICE_CHECK_ERROR();
}

void sort_tracks_by_action_id(
    CoreStateData<Ownership::reference, MemSpace::device> const& states)
{
    CELER_EXPECT(states.size() > 0);
    Span track_slots{
        states.track_slots[AllItems<TrackSlotId::size_type, MemSpace::device>{}]};
    thrust::sort(
        thrust::device,
        thrust::device_pointer_cast(track_slots.begin()),
        thrust::device_pointer_cast(track_slots.end()),
        step_limit_comparator{
            states.sim.step_limit[AllItems<StepLimit, MemSpace::device>{}]});
    CELER_DEVICE_CHECK_ERROR();
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
