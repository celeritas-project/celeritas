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
    CELER_FUNCTION bool operator()(TrackStatus const& track_status) const
    {
        return track_status == TrackStatus::alive;
    }
};
}  // namespace

template<>
void partition_tracks_by_status(
    CoreStateData<Ownership::reference, MemSpace::device> const& states)
{
    CELER_EXPECT(states.size() > 0);
    Span track_slots{
        states.track_slots[AllItems<TrackSlotId::size_type, MemSpace::device>{}]};
    Span status{states.sim.status[AllItems<TrackStatus, MemSpace::device>{}]};
    thrust::partition(thrust::device,
                      thrust::device_pointer_cast(track_slots.begin()),
                      thrust::device_pointer_cast(track_slots.end()),
                      thrust::device_pointer_cast(status.data()),
                      alive_predicate{});
    CELER_DEVICE_CHECK_ERROR();
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
