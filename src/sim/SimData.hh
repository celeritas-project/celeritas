//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simulation state of a track.
 */
struct SimTrackState
{
    TrackId   track_id;      //!< Unique ID for this track
    TrackId   parent_id;     //!< ID of parent that created it
    EventId   event_id;      //!< ID of originating event
    size_type num_steps = 0; //!< Total number of steps taken
    bool      alive = false; //!< Whether this track is alive
};

using SimTrackInitializer = SimTrackState;

//---------------------------------------------------------------------------//
/*!
 * Data storage/access for simulation states.
 */
template<Ownership W, MemSpace M>
struct SimStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    Items<SimTrackState> state;

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const { return !state.empty(); }

    //! State size
    CELER_FUNCTION ThreadId::size_type size() const { return state.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimStateData& operator=(SimStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        state = other.state;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize simulation states and set \c alive to be false.
 */
template<MemSpace M>
void resize(SimStateData<Ownership::value, M>* data, size_type size)
{
    CELER_EXPECT(size > 0);
    StateCollection<SimTrackState, Ownership::value, MemSpace::host> state;
    std::vector<SimTrackState> initial_state(size);
    make_builder(&state).insert_back(initial_state.begin(),
                                     initial_state.end());
    data->state = state;
    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
