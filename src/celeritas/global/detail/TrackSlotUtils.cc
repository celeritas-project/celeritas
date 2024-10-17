//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/TrackSlotUtils.cc
//---------------------------------------------------------------------------//
#include "TrackSlotUtils.hh"

#include <algorithm>
#include <random>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Shuffle track slot indices.
 */
void shuffle_track_slots(
    Collection<TrackSlotId::size_type, Ownership::value, MemSpace::host, ThreadId>*
        track_slots,
    StreamId)
{
    CELER_EXPECT(track_slots);
    auto* start = track_slots->data().get();
    auto seed = static_cast<unsigned int>(track_slots->size());
    std::mt19937 g{seed};
    std::shuffle(start, start + track_slots->size(), g);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
