//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/TrackSortUtils.cc
//---------------------------------------------------------------------------//
#include "TrackSortUtils.hh"

#include <algorithm>
#include <numeric>
#include <random>

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
void fill_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots)
{
    std::iota(track_slots.data(), track_slots.data() + track_slots.size(), 0);
}

/*!
 * Shuffle track slots
 */
template<>
void shuffle_track_slots<MemSpace::host>(Span<TrackSlotId::size_type> track_slots)
{
    unsigned int seed = track_slots.size();
    std::mt19937 g{seed};
    std::shuffle(track_slots.begin(), track_slots.end(), g);
}

template<>
void partition_tracks_by_status(
    CoreStateData<Ownership::reference, MemSpace::host> const& states)
{
    CELER_EXPECT(states.size() > 0);
    Span track_slots{
        states.track_slots[AllItems<TrackSlotId::size_type, MemSpace::host>{}]};
    std::partition(track_slots.begin(),
                   track_slots.end(),
                   [&status = states.sim.status](auto const track_slot) {
                       return status[TrackSlotId{track_slot}]
                              == TrackStatus::alive;
                   });
}

template<>
void sort_tracks_by_action_id(
    CoreStateData<Ownership::reference, MemSpace::host> const& states)
{
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
