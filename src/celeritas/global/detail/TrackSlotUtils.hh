//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/TrackSlotUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Shuffle track slot indices
void shuffle_track_slots(
    Collection<TrackSlotId::size_type, Ownership::value, MemSpace::host, ThreadId>*,
    StreamId);
void shuffle_track_slots(
    Collection<TrackSlotId::size_type, Ownership::value, MemSpace::device, ThreadId>*,
    StreamId);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline void shuffle_track_slots(
    Collection<TrackSlotId::size_type, Ownership::value, MemSpace::device, ThreadId>*,
    StreamId)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
