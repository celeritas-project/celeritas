//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Particle-dependent parameters for killing looping tracks.
 *
 * Tracks that are flagged as looping (taking a large number of substeps in the
 * field propagator) will be killed immediately if their energy is below \c
 * threshold_energy or after \c max_steps step iterations if their energy is
 * above the threshold. The \c threshold_energy is equivalent to the "important
 * energy" in Geant4.
 */
struct LoopingThreshold
{
    using Energy = units::MevEnergy;

    size_type max_steps{10};
    Energy threshold_energy{250};

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const { return max_steps > 0; }
};
//---------------------------------------------------------------------------//
/*!
 * Persistent simulation data.
 */
template<Ownership W, MemSpace M>
struct SimParamsData
{
    //// TYPES ////

    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;

    //// DATA ////

    ParticleItems<LoopingThreshold> looping;

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const { return !looping.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimParamsData& operator=(SimParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        looping = other.looping;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Simulation state of a track.
 *
 * TODO change to struct-of-arrays
 */
struct SimTrackState
{
    TrackId track_id;  //!< Unique ID for this track
    TrackId parent_id;  //!< ID of parent that created it
    EventId event_id;  //!< ID of originating event
    size_type num_steps{0};  //!< Total number of steps taken
    size_type num_looping_steps{0};  //!< Number of steps taken since the
                                     //!< track was flagged as looping
    real_type time{0};  //!< Time elapsed in lab frame since start of event [s]

    TrackStatus status{TrackStatus::inactive};
    StepLimit step_limit;
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
    CELER_FUNCTION TrackSlotId::size_type size() const { return state.size(); }

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
 *
 * TODO: replace with resize + fill
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
}  // namespace celeritas
