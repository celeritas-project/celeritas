//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackSortUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"

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
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<>
inline void fill_track_slots<MemSpace::device>(Span<TrackSlotId::size_type>)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
