//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitInterface.cc
//---------------------------------------------------------------------------//
#include "TrackInitInterface.hh"

#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"
#include "comm/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize and initialize track initializer data on device.
 */
void resize(
    TrackInitStateData<Ownership::value, MemSpace::device>* data,
    const TrackInitParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(celeritas::device());

    // Allocate device data
    auto capacity = params.storage_factor * size;
    make_builder(&data->initializers.storage).resize(capacity);
    make_builder(&data->parents.storage).resize(capacity);
    make_builder(&data->secondary_counts).resize(size);

    // Start with an empty vector of track initializers and parent thread IDs
    data->initializers.resize(0);
    data->parents.resize(0);

    // Initialize vacancies to mark all track slots as empty
    StateCollection<size_type, Ownership::value, MemSpace::host> vacancies;
    make_builder(&vacancies).resize(size);
    for (auto i : range(ThreadId{size}))
        vacancies[i] = i.get();
    data->vacancies.storage = vacancies;
    data->vacancies.resize(size);

    // Initialize the track counter for each event as the number of primary
    // particles in that event
    std::vector<size_type> counters;
    for (const auto& p : params.primaries[AllItems<Primary, MemSpace::host>{}])
    {
        const auto event_id = p.event_id.get();
        if (!(event_id < counters.size()))
        {
            counters.resize(event_id + 1);
        }
        ++counters[event_id];
    }
    Collection<TrackId::size_type, Ownership::value, MemSpace::host, EventId>
        track_counters;
    make_builder(&track_counters).insert_back(counters.begin(), counters.end());
    data->track_counters = track_counters;
    data->num_primaries  = params.primaries.size();
}

//---------------------------------------------------------------------------//
/*!
 * Resize and initialize track initializer data on host.
 */
void resize(
    TrackInitStateData<Ownership::value, MemSpace::host>*,
    const TrackInitParamsData<Ownership::const_reference, MemSpace::host>&,
    size_type)
{
    CELER_NOT_IMPLEMENTED("Host track initializer state");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
