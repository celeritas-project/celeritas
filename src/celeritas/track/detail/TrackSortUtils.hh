//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

// Initialize default threads to track_slots mapping, track_slots[i] = i
template<MemSpace M,
         typename Size,
         typename = std::enable_if_t<std::is_unsigned_v<Size>>>
void fill_track_slots(Span<Size> track_slots);

template<>
void fill_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots);
template<>
void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type> track_slots);

//---------------------------------------------------------------------------//
// Shuffle tracks
template<MemSpace M,
         typename Size,
         typename = std::enable_if_t<std::is_unsigned_v<Size>>>
void shuffle_track_slots(Span<Size> track_slots);

template<>
void shuffle_track_slots<MemSpace::host>(
    Span<TrackSlotId::size_type> track_slots);
template<>
void shuffle_track_slots<MemSpace::device>(
    Span<TrackSlotId::size_type> track_slots);

//---------------------------------------------------------------------------//
// Sort tracks

void partition_tracks_by_status(
    CoreStateData<Ownership::reference, MemSpace::host> const& states);

void partition_tracks_by_status(
    CoreStateData<Ownership::reference, MemSpace::device> const& states);

//---------------------------------------------------------------------------//

void sort_tracks_by_action_id(
    CoreStateData<Ownership::reference, MemSpace::host> const& states);

void sort_tracks_by_action_id(
    CoreStateData<Ownership::reference, MemSpace::device> const& states);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<>
inline void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<>
inline void shuffle_track_slots<MemSpace::device>(Span<TrackSlotId::size_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void partition_tracks_by_status(
    CoreStateData<Ownership::reference, MemSpace::device> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void sort_tracks_by_action_id(
    CoreStateData<Ownership::reference, MemSpace::device> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
