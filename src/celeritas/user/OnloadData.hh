//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/OnloadData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/PinnedAllocator.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared attributes about the onloaded steps being collected.
 */
template<Ownership W, MemSpace M>
struct OnloadParamsData
{
    //// DATA ////

    //! Filter out steps that undergo onloading processes
    bool gamma_nuclear{true};
    bool electro_nuclear{true};

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return (gamma_nuclear || electro_nuclear);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OnloadParamsData& operator=(OnloadParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        gamma_nuclear = other.gamma_nuclear;
        electro_nuclear = other.electro_nuclear;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Onload state data for a single step for many tracks in parallel.
 */
template<Ownership W, MemSpace M>
struct OnloadStateDataImpl
{
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    using Energy = units::MevEnergy;

    //// DATA ////

    //! Step (post-step) data
    StateItems<Energy> energy;
    StateItems<real_type> time;
    StateItems<real_type> weight;
    StateItems<Real3> pos;
    StateItems<Real3> dir;

    //! Track ID is always assigned (but will be false for inactive tracks)
    StateItems<TrackId> track_id;

    // Physics
    StateItems<ParticleId> particle_id;

    //// METHODS ////

    //! True if constructed
    explicit CELER_FUNCTION operator bool() const
    {
        return !track_id.empty() && !particle_id.empty();
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return track_id.size();
    }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    OnloadStateDataImpl& operator=(OnloadStateDataImpl<W2, M2>& other)
    {
        CELER_EXPECT(other || (M == MemSpace::host && other.size() == 0));

        energy = other.energy;
        time = other.time;
        weight = other.weight;
        pos = other.pos;
        dir = other.dir;
        track_id = other.track_id;
        particle_id = other.particle_id;

        return *this;
    }
};

template<Ownership W, MemSpace M>
struct OnloadStateData
{
    //// TYPES ////
    using StepDataImpl = OnloadStateDataImpl<W, M>;
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    //! Gathered data for a single step
    StepDataImpl data;

    //! Scratch space for gathering the data on device based on track validity
    StepDataImpl scratch;

    //! Thread IDs of active tracks that are in a detector
    StateItems<size_type> valid_id;

    //! Unique identifier for "thread-local" data.
    StreamId stream_id;

    //// METHODS ////

    //! True if constructed and correctly sized
    explicit CELER_FUNCTION operator bool() const
    {
        auto right_sized = [this](auto const& t) {
            return (t.size() == this->size())
                   || (t.size() == 0 && M == MemSpace::host);
        };

        return data.size() > 0 && right_sized(scratch) && right_sized(valid_id)
               && stream_id;
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const { return data.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    OnloadStateData& operator=(OnloadStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        data = other.data;
        scratch = other.scratch;
        valid_id = other.valid_id;
        stream_id = other.stream_id;
        return *this;
    }
};

template<MemSpace M>
struct OnloadState
{
    //! Host pointer to externally owned onload data reference
    OnloadStateData<Ownership::reference, M> const& steps;

    //! Stream ID (local to each core state)
    StreamId stream_id;
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Resize the onload step data.
 */
template<MemSpace M>
inline void resize(OnloadStateDataImpl<Ownership::value, M>* state,
                   size_type size)
{
    CELER_EXPECT(state->size() == 0);
    CELER_EXPECT(size > 0);

    resize(&state->energy, size);
    resize(&state->time, size);
    resize(&state->weight, size);
    resize(&state->pos, size);
    resize(&state->dir, size);
    resize(&state->track_id, size);
    resize(&state->particle_id, size);
}

//---------------------------------------------------------------------------//
/*!
 * Resize the onload state data.
 */
template<MemSpace M>
inline void resize(OnloadStateData<Ownership::value, M>* state,
                   HostCRef<OnloadParamsData> const& params,
                   StreamId stream_id,
                   size_type size)
{
    CELER_EXPECT(state->size() == 0);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(params);

    state->stream_id = stream_id;

    resize(&state->data, size);

    if constexpr (M == MemSpace::device)
    {
        // Allocate extra space on device for gathering step data
        resize(&state->scratch, size);
        resize(&state->valid_id, size);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
