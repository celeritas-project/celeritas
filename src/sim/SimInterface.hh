//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "detail/SimStateInit.hh"
#include "Types.hh"

#ifndef __CUDA_ARCH__
#    include "base/CollectionBuilder.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Static properties of the simulation.
 *
 * There is no persistent data needed on device or at runtime: the params are
 * only used for construction.
 */
template<Ownership W, MemSpace M>
struct SimParamsData
{
    /* no data on device */

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimParamsData& operator=(const SimParamsData<W2, M2>&)
    {
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Simulation state of a track.
 */
struct SimTrackState
{
    TrackId track_id;      //!< Unique ID for this track
    TrackId parent_id;     //!< ID of parent that created it
    EventId event_id;      //!< ID of originating event
    bool    alive = false; //!< Whether this track is alive
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

#ifndef __CUDA_ARCH__
//---------------------------------------------------------------------------//
/*!
 * Resize simulation states and set \c alive to be false.
 */
template<MemSpace M>
void resize(SimStateData<Ownership::value, M>* data,
            const SimParamsData<Ownership::const_reference, MemSpace::host>&,
            size_type size)
{
    CELER_EXPECT(size > 0);
    make_builder(&data->state).resize(size);
    detail::sim_state_init(make_ref(*data));
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
