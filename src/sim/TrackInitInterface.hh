//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackInitInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Types.hh"
#include "geometry/GeoInterface.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/base/Primary.hh"
#include "SimInterface.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Static track initializer data.
 *
 * There is no persistent data needed on device. Primaries are copied to device
 * only when they are needed to initialize new tracks and are not stored on
 * device. \c storage_factor is only used at construction to allocate memory
 * for track initializers and parent track IDs.
 */
template<Ownership W, MemSpace M>
struct TrackInitParamsData;

template<Ownership W>
struct TrackInitParamsData<W, MemSpace::device>
{
    /* no data on device */

    //// METHODS ////

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitParamsData& operator=(const TrackInitParamsData<W2, M2>&)
    {
        return *this;
    }
};

template<Ownership W>
struct TrackInitParamsData<W, MemSpace::host>
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, MemSpace::host>;

    //// DATA ////

    Items<Primary> primaries;     //!< Primary particles
    size_type storage_factor = 3; //!< Initializer/parent storage per track

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !primaries.empty() && storage_factor > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitParamsData& operator=(const TrackInitParamsData<W2, M2>& other)
    {
        primaries      = other.primaries;
        storage_factor = other.storage_factor;
        return *this;
    }
};

using TrackInitParamsHostRef
    = TrackInitParamsData<Ownership::const_reference, MemSpace::host>;

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
    CELER_FUNCTION SpanT pointers()
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
 * most cases accessed by \c ThreadId. Specifically, \c initializers, \c
 * parents, and \c vacancies are all resizable, and \c track_counters has size
 * \c num_events.
 * - \c initializers stores the data for primaries and secondaries waiting to
 *   be turned into new tracks and can be any size up to \c storage_factor * \c
 *   num_tracks.
 * - \c parents is the \c ThreadId of the parent tracks of the initializers.
 * - \c vacancies stores the indices of the threads of tracks that have been
 *   killed; the size will be <= the number of tracks.
 * - \c track_counters stores the total number of particles that have been
 *   created per event.
 * - \c secondary_counts stores the number of secondaries created by each track
 *   (and is the only true state data).
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
    ResizableItems<ThreadId>         parents;
    ResizableItems<size_type>        vacancies;
    StateItems<size_type>            secondary_counts;
    EventItems<TrackId::size_type>   track_counters;

    size_type num_primaries{}; //!< Number of uninitialized primaries

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return initializers && parents && vacancies
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
        num_primaries    = other.num_primaries;
        return *this;
    }
};

using TrackInitStateDeviceRef
    = TrackInitStateData<Ownership::reference, MemSpace::device>;
using TrackInitStateDeviceVal
    = TrackInitStateData<Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
// Resize and initialize track initializer data.
template<MemSpace M>
inline void
resize(TrackInitStateData<Ownership::value, M>*,
       const TrackInitParamsData<Ownership::const_reference, MemSpace::host>&,
       size_type);

//---------------------------------------------------------------------------//

} // namespace celeritas

#include "sim/TrackInitInterface.i.hh"
