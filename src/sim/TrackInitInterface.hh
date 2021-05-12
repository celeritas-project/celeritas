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
 * There is no persistent data needed on device or at runtime: the params are
 * only used for construction.
 */
template<Ownership W, MemSpace M>
struct TrackInitParamsData;

template<Ownership W>
struct TrackInitParamsData<W, MemSpace::device>
{
    /* no data on device */

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

    Items<Primary> primaries;

    size_type storage_factor = 3; //!< Initializer/parent storage per tracks

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
 * Storage for data used to initialize new tracks.
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
// Resize and initialize track initializer data on device.
void resize(
    TrackInitStateData<Ownership::value, MemSpace::device>*,
    const TrackInitParamsData<Ownership::const_reference, MemSpace::host>&,
    size_type);

//---------------------------------------------------------------------------//
// Resize and initialize track initializer data on host (not implemented).
void resize(
    TrackInitStateData<Ownership::value, MemSpace::host>*,
    const TrackInitParamsData<Ownership::const_reference, MemSpace::host>&,
    size_type);

//---------------------------------------------------------------------------//
} // namespace celeritas
