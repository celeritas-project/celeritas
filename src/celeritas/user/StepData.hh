//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
enum class StepPoint
{
    pre,
    post,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Which track properties to gather at every step.
 */
struct StepSelection
{
    bool pre_step{true};
    bool sim{true};
    bool geo{true};
    bool phys{true};
};

//---------------------------------------------------------------------------//
/*!
 * Shared attributes about the hits being collected.
 *
 * This will be expanded to include filters for particle type, region, etc.
 */
template<Ownership W, MemSpace M>
struct StepParamsData
{
    //// DATA ////

    StepSelection selection;

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StepParamsData& operator=(const StepParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        selection = other.selection;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Gathered state data for beginning/end of step data for tracks in parallel.
 *
 * - Depending on the collection options, some of these state data may be
 * empty.
 * - If a track is outside the volume (which can only happen at the end-of-step
 *   evaluation) the VolumeId will be "false".
 */
template<Ownership W, MemSpace M>
struct StepPointStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    using Energy     = units::MevEnergy;

    // Sim
    StateItems<real_type> time;

    // Geo
    StateItems<Real3>    pos;
    StateItems<Real3>    dir;
    StateItems<VolumeId> volume;

    // Physics
    StateItems<Energy> energy;

    //// METHODS ////

    //! Always true since we all of the data could be empty
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    StepPointStateData& operator=(StepPointStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        time   = other.time;
        pos    = other.pos;
        dir    = other.dir;
        volume = other.volume;
        energy = other.energy;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Gathered data for a single step for many tracks in parallel.
 *
 * - The track ID will be set to "false" if the track is inactive.
 * - Depending on the collection options, some of these state data may be
 * empty.
 */
template<Ownership W, MemSpace M>
struct StepStateData
{
    //// TYPES ////

    using StepPointData = StepPointStateData<W, M>;
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    using Energy     = units::MevEnergy;

    //// DATA ////

    // Pre- and post-step data
    EnumArray<StepPoint, StepPointData> points;

    // Track ID is always set
    StateItems<TrackId> track;

    // Sim
    StateItems<EventId>   event;
    StateItems<size_type> track_step_count;
    StateItems<ActionId>  action;
    StateItems<real_type> step_length;

    // Physics
    StateItems<ParticleId> particle;
    StateItems<Energy>     energy_deposition;

    //// METHODS ////

    //! True if constructed and correctly sized
    explicit CELER_FUNCTION operator bool() const
    {
        auto right_sized = [this](const auto& t) {
            return (t.size() == this->size()) || t.empty();
        };

        return !track.empty() && right_sized(event)
               && right_sized(track_step_count) && right_sized(action)
               && right_sized(step_length) && right_sized(particle)
               && right_sized(energy_deposition);
    }

    //! State size
    CELER_FUNCTION size_type size() const { return track.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    StepStateData& operator=(StepStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        track                   = other.track;
        points[StepPoint::pre]  = other.points[StepPoint::pre];
        points[StepPoint::post] = other.points[StepPoint::post];
        event                   = other.event;
        track_step_count        = other.track_step_count;
        action                  = other.action;
        step_length             = other.step_length;
        particle                = other.particle;
        energy_deposition       = other.energy_deposition;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize a state point.
 */
template<MemSpace M>
inline void resize(StepPointStateData<Ownership::value, M>* state,
                   const HostCRef<StepParamsData>&          params,
                   size_type                                size)
{
    CELER_EXPECT(size > 0);
    if (params.selection.sim)
    {
        resize(&state->time, size);
    }
    if (params.selection.geo)
    {
        resize(&state->pos, size);
        resize(&state->dir, size);
        resize(&state->volume, size);
    }
    if (params.selection.phys)
    {
        resize(&state->energy, size);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Resize the state.
 */
template<MemSpace M>
inline void resize(StepStateData<Ownership::value, M>* state,
                   const HostCRef<StepParamsData>&     params,
                   size_type                           size)
{
    CELER_EXPECT(state->size() == 0);
    CELER_EXPECT(size > 0);

    if (params.selection.pre_step)
    {
        resize(&state->points[StepPoint::pre], params, size);
    }
    resize(&state->points[StepPoint::post], params, size);

    resize(&state->track, size);
    if (params.selection.sim)
    {
        resize(&state->event, size);
        resize(&state->track_step_count, size);
        resize(&state->step_length, size);
        resize(&state->action, size);
    }
    if (params.selection.phys)
    {
        resize(&state->particle, size);
        resize(&state->energy_deposition, size);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
