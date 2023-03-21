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
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include "corecel/Macros.hh"

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
    std::mt19937 g{track_slots.size()};
    thrust::shuffle(
        thrust::device_pointer_cast(track_slots.data()),
        thrust::device_pointer_cast(track_slots.data() + track_slots.size()),
        g);
    CELER_DEVICE_CHECK_ERROR();
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
