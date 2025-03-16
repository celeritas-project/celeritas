//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/TrackSlotUtils.cu
//---------------------------------------------------------------------------//
#include "TrackSlotUtils.hh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include "corecel/sys/Thrust.device.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Shuffle track slot indices.
 */
void shuffle_track_slots(
    Collection<TrackSlotId::size_type, Ownership::value, MemSpace::device, ThreadId>*
        track_slots,
    StreamId stream)
{
    CELER_EXPECT(track_slots);
    using result_type = thrust::default_random_engine::result_type;
    thrust::default_random_engine g{
        static_cast<result_type>(track_slots->size())};
    auto start = thrust::device_pointer_cast(track_slots->data().get());
    thrust::shuffle(
        thrust_execute_on(stream), start, start + track_slots->size(), g);
    CELER_DEVICE_API_CALL(PeekAtLastError());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
