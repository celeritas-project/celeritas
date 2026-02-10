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
#include "celeritas/track/CoreStateCounters.hh"

#include "TrackInitializer.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Storage for dynamic data used to initialize new optical photon tracks.
 *
 * - \c initializers stores the data for track initializers and secondaries
 *   waiting to be turned into new tracks and can be any size up to \c
 *   capacity.
 * - \c vacancies stores the \c TrackSlotid of the tracks that have been
 *   killed; the size will be <= the number of track states.
 * - \c counters stores the number of tracks with a given status and is updated
 *   during each step of the simulation of an event.
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

    StateItems<TrackSlotId> vacancies;

    // Maintain the counters here to allow device-resident computation with
    // synchronization between host and device only at the end of a step or
    // when explicitly requested, such as in the tests
    Items<CoreStateCounters> counters;

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !vacancies.empty() && !counters.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrackInitStateData& operator=(TrackInitStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        vacancies = other.vacancies;
        counters = other.counters;
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
            StreamId stream,
            size_type size)
{
    CELER_EXPECT(size > 0);

    // Initialize vacancies to mark all track slots as empty
    resize(&data->vacancies, size);
    fill_sequence(&data->vacancies, stream);

    // Initialize the counters for the step to zero
    resize(&data->counters, 1);
    fill(CoreStateCounters{}, &data->counters);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//

}  // namespace optical
}  // namespace celeritas
