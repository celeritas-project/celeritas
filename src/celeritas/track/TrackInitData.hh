//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/Primary.hh"

#include "SimData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Persistent data for track initialization.
 *
 * TODO: change \c max_events to be the maximum number of events in flight at
 * once rather than the maximum number of events that can be run over the
 * entire simulation
 */
template<Ownership W, MemSpace M>
struct TrackInitParamsData
{
    size_type capacity{0};  //!< Track initializer storage size
    size_type max_events{0};  //!< Maximum number of events that can be run
    TrackOrder track_order{TrackOrder::unsorted};  //!< How to sort tracks on
                                                   //!< gpu

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return capacity > 0 && max_events > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitParamsData& operator=(TrackInitParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        capacity = other.capacity;
        max_events = other.max_events;
        track_order = other.track_order;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Lightweight version of a track used to initialize new tracks from primaries
 * or secondaries.
 */
struct TrackInitializer
{
    SimTrackInitializer sim;
    GeoTrackInitializer geo;
    ParticleTrackInitializer particle;

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return sim && geo && particle;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Storage for dynamic data used to initialize new tracks.
 *
 * Not all of this is technically "state" data, though it is all mutable and in
 * most cases accessed by \c TrackSlotId. Specifically, \c initializers and \c
 * vacancies are resizable, and \c track_counters has size
 * \c max_events.
 * - \c initializers stores the data for primaries and secondaries waiting to
 *   be turned into new tracks and can be any size up to \c capacity.
 * - \c parents is the \c TrackSlotId of the parent tracks of the initializers.
 * - \c vacancies stores the \c TrackSlotid of the tracks that have been
 *   killed; the size will be <= the number of track states.
 * - \c track_counters stores the total number of particles that have been
 *   created per event.
 * - \c secondary_counts stores the number of secondaries created by each track
 *   (with one remainder at the end for storing the accumulated number of
 *   secondaries)
 */
template<Ownership W, MemSpace M>
struct TrackInitStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;
    template<class T>
    using EventItems = Collection<T, W, M, EventId>;
    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    StateItems<TrackSlotId> parents;
    StateItems<size_type> secondary_counts;
    StateItems<TrackSlotId> vacancies;
    EventItems<TrackId::size_type> track_counters;

    // Storage (size is "capacity", not "currently used": see
    // CoreStateCounters)
    Items<TrackInitializer> initializers;

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !parents.empty() && secondary_counts.size() == parents.size() + 1
               && !track_counters.empty() && vacancies.size() == parents.size()
               && !initializers.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitStateData& operator=(TrackInitStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        parents = other.parents;
        secondary_counts = other.secondary_counts;
        track_counters = other.track_counters;

        vacancies = other.vacancies;
        initializers = other.initializers;

        return *this;
    }
};

using TrackInitStateDeviceRef = DeviceRef<TrackInitStateData>;
using TrackInitStateHostRef = HostRef<TrackInitStateData>;

//---------------------------------------------------------------------------//
/*!
 * Resize and initialize track initializer data.
 *
 * Here \c size is the number of track states, and the "capacity" is the
 * maximum number of track initializers (inactive/pending tracks) that we can
 * hold.
 *
 * \warning It's likely that for GPU runs, the capacity should be greater than
 * the size, but that might not be the case universally, so it is not asserted.
 */
template<MemSpace M>
void resize(TrackInitStateData<Ownership::value, M>* data,
            HostCRef<TrackInitParamsData> const& params,
            size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(M == MemSpace::host || celeritas::device());

    // Allocate device data
    resize(&data->parents, size);
    resize(&data->secondary_counts, size + 1);
    resize(&data->track_counters, params.max_events);

    // Initialize the track counter for each event to zero
    fill(size_type(0), &data->track_counters);

    // Initialize vacancies to mark all track slots as empty
    StateCollection<TrackSlotId, Ownership::value, MemSpace::host> vacancies;
    resize(&vacancies, size);
    for (auto i : range(size))
    {
        vacancies[TrackSlotId{i}] = TrackSlotId{i};
    }
    data->vacancies = std::move(vacancies);

    // Reserve space for initializers
    resize(&data->initializers, params.capacity);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//

}  // namespace celeritas
