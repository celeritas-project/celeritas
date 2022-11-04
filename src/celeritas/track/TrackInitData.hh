//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInitData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ThreadId.hh"
#include "orange/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/Primary.hh"

#include "SimData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Persistent data for track initialization.
 */
template<Ownership W, MemSpace M>
struct TrackInitParamsData
{
    size_type capacity{0};   //!< Track initializer storage size
    size_type max_events{0}; //!< Maximum number of events that can be run

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return capacity > 0 && max_events > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitParamsData& operator=(const TrackInitParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        capacity   = other.capacity;
        max_events = other.max_events;
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
    SimTrackInitializer      sim;
    GeoTrackInitializer      geo;
    ParticleTrackInitializer particle;
};

//---------------------------------------------------------------------------//
/*!
 * StateCollection with a fixed capacity and dynamic size.
 */
template<class T, Ownership W, MemSpace M>
struct ResizableData
{
    //// TYPES ////

    using CollectionT    = StateCollection<T, W, M>;
    using ItemIdT        = typename CollectionT::ItemIdT;
    using ItemRangeT     = typename CollectionT::ItemRangeT;
    using reference_type = typename CollectionT::reference_type;
    using SpanT          = typename CollectionT::SpanT;

    //// DATA ////

    CollectionT storage;
    size_type   count{};

    //// METHODS ////

    // Whether the interface is initialized
    explicit inline CELER_FUNCTION operator bool() const
    {
        return !storage.empty();
    }

    //! Capacity of allocated storage
    CELER_FUNCTION size_type capacity() const { return storage.size(); }

    //! Number of elements
    CELER_FUNCTION size_type size() const { return count; }

    //! Change the size without changing capacity
    CELER_FUNCTION void resize(size_type size)
    {
        CELER_EXPECT(size <= this->capacity());
        count = size;
    }

    //! Access a single element
    CELER_FUNCTION reference_type operator[](ItemIdT i) const
    {
        CELER_EXPECT(i < this->size());
        return storage[i];
    }

    //! View to the data
    CELER_FUNCTION SpanT data()
    {
        return storage[ItemRangeT{ItemIdT{0}, ItemIdT{this->size()}}];
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ResizableData& operator=(ResizableData<T, W2, M2>& other)
    {
        CELER_EXPECT(other);
        storage = other.storage;
        count   = other.count;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Storage for dynamic data used to initialize new tracks.
 *
 * Not all of this is technically "state" data, though it is all mutable and in
 * most cases accessed by \c ThreadId. Specifically, \c initializers and \c
 * vacancies are resizable, and \c track_counters has size
 * \c max_events.
 * - \c initializers stores the data for primaries and secondaries waiting to
 *   be turned into new tracks and can be any size up to \c capacity.
 * - \c parents is the \c ThreadId of the parent tracks of the initializers.
 * - \c vacancies stores the indices of the threads of tracks that have been
 *   killed; the size will be <= the number of tracks.
 * - \c track_counters stores the total number of particles that have been
 *   created per event.
 * - \c secondary_counts stores the number of secondaries created by each track
 */
template<Ownership W, MemSpace M>
struct TrackInitStateData
{
    //// TYPES ////

    template<class T>
    using EventItems = Collection<T, W, M, EventId>;
    template<class T>
    using ResizableItems = ResizableData<T, W, M>;
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    ResizableItems<TrackInitializer> initializers;
    ResizableItems<size_type>        vacancies;
    StateItems<ThreadId>             parents;
    StateItems<size_type>            secondary_counts;
    EventItems<TrackId::size_type>   track_counters;

    size_type num_secondaries{}; //!< Number of secondaries produced in a step

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return initializers && vacancies && !parents.empty()
               && !secondary_counts.empty() && !track_counters.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitStateData& operator=(TrackInitStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        initializers     = other.initializers;
        parents          = other.parents;
        vacancies        = other.vacancies;
        secondary_counts = other.secondary_counts;
        track_counters   = other.track_counters;
        num_secondaries  = other.num_secondaries;
        return *this;
    }
};

using TrackInitStateDeviceRef = DeviceRef<TrackInitStateData>;
using TrackInitStateHostRef   = HostRef<TrackInitStateData>;

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
            const HostCRef<TrackInitParamsData>&     params,
            size_type                                size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(M == MemSpace::host || celeritas::device());

    // Allocate device data
    resize(&data->initializers.storage, params.capacity);
    resize(&data->parents, size);
    resize(&data->secondary_counts, size);
    resize(&data->track_counters, params.max_events);

    // Start with an empty vector of track initializers
    data->initializers.resize(0);

    // Initialize vacancies to mark all track slots as empty
    StateCollection<size_type, Ownership::value, MemSpace::host> vacancies;
    resize(&vacancies, size);
    for (auto i : range(size))
    {
        vacancies[ThreadId{i}] = i;
    }
    data->vacancies.storage = vacancies;
    data->vacancies.resize(size);

    // Initialize the track counter for each event to zero
    fill(size_type(0), &data->track_counters);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//

} // namespace celeritas
