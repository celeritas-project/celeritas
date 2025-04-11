//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/TrackInitData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

#include "TrackInitializer.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Persistent data for optical track initialization.
 */
template<Ownership W, MemSpace M>
struct TrackInitParamsData
{
    size_type capacity{0};  //!< Optical primary buffer storage size

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const { return capacity > 0; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitParamsData& operator=(TrackInitParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        capacity = other.capacity;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Storage for dynamic data used to initialize new optical photon tracks.
 *
 * - \c initializers stores the data for track initializers and secondaries
 *   waiting to be turned into new tracks and can be any size up to \c
 *   capacity.
 * - \c vacancies stores the \c TrackSlotid of the tracks that have been
 *   killed; the size will be <= the number of track states.
 */
template<Ownership W, MemSpace M>
struct TrackInitStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;
    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    Items<TrackInitializer> initializers;
    StateItems<TrackSlotId> vacancies;

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !initializers.empty() && !vacancies.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitStateData& operator=(TrackInitStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        initializers = other.initializers;
        vacancies = other.vacancies;

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize and initialize data.
 *
 * Here \c size is the number of track states, and the "capacity" is the
 * maximum number of initializers that can be buffered.
 */
template<MemSpace M>
void resize(TrackInitStateData<Ownership::value, M>* data,
            HostCRef<TrackInitParamsData> const& params,
            StreamId stream,
            size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);

    resize(&data->initializers, params.capacity);
    resize(&data->vacancies, size);

    // Initialize vacancies to mark all track slots as empty
    fill_sequence(&data->vacancies, stream);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//

}  // namespace optical
}  // namespace celeritas
