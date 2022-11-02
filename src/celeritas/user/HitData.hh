//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/HitData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Which track properties to gather at every step.
 */
struct HitSelection
{
    bool sim{true};
    bool geo{true};
    bool phys{true};
    bool post_step{true};
};

//---------------------------------------------------------------------------//
/*!
 * Shared attributes about the hits being collected.
 *
 * This will be expanded to include filters for particle type, region, etc.
 */
template<Ownership W, MemSpace M>
struct HitParamsData
{
    //// DATA ////

    HitSelection selection;
    bool         is_post_step{};

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParticleParamsData& operator=(const ParticleParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        selection    = other.selection;
        is_post_step = other.is_post_step;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Gathered data for a single "state" (pre or post) for many tracks in
 * parallel.
 *
 * - The track ID will be set to "false" if the track is inactive.
 * - The post-step data and energy deposition are *only* valid for post-step
 *   callbacks.
 * - Depending on the collection options, some of these state data may be
 * empty.
 * - If a track is outside the volume (which can only happen at the end-of-step
 *   evaluation) the VolumeId will be "false".
 */
template<Ownership W, MemSpace M>
struct HitStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    using Energy     = units::MevEnergy;

    //// DATA ////

    // Always on
    StateItems<TrackId> track;

    // Sim
    StateItems<EventId>   event;
    StateItems<size_type> num_steps;
    StateItems<real_type> time;

    // Geo
    StateItems<Real3>    pos;
    StateItems<Real3>    dir;
    StateItems<VolumeId> volume;

    // Physics
    StateItems<ParticleId> particle;
    StateItems<Energy>     kinetic_energy;
    StateItems<Energy>     energy_deposition;

    // Post-step
    StateItems<real_type> step_length; // "true" step
    StateItems<ActionId>  action;

    //// METHODS ////

    //! True if constructed and correctly sized
    explicit CELER_FUNCTION operator bool() const
    {
        return !track.empty() && (event.size() == track.size() || event.empty())
               && (num_steps.size() == track.size() || num_steps.empty())
               && (time.size() == track.size() || time.empty())
               && (pos.size() == track.size() || pos.empty())
               && (dir.size() == track.size() || dir.empty())
               && (volume.size() == track.size() || volume.empty())
               && (particle.size() == track.size() || particle.empty())
               && (kinetic_energy.size() == track.size()
                   || kinetic_energy.empty())
               && (energy_deposition.size() == track.size()
                   || energy_deposition.empty())
               && (step_length.size() == track.size() || step_length.empty())
               && (action.size() == track.size() || action.empty());
    }

    //! State size
    CELER_FUNCTION size_type size() const { return state.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    HitStateData& operator=(HitStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        track             = other.track;
        event             = other.event;
        num_steps         = other.num_steps;
        time              = other.time;
        pos               = other.pos;
        dir               = other.dir;
        volume            = other.volume;
        particle          = other.particle;
        kinetic_energy    = other.kinetic_energy;
        energy_deposition = other.energy_deposition;
        step_length       = other.step_length;
        action            = other.action;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize the state in host code.
 */
template<MemSpace M>
inline void resize(HitStateData<Ownership::value, M>* state,
                   const HostCRef<HitParamsData>&     params,
                   size_type                          size)
{
    CELER_EXPECT(size > 0);
    resize(&state->track, size);
    if (params.scalars.sim)
    {
        resize(&state->event, size);
        resize(&state->num_steps, size);
        resize(&state->time, size);
    }
    if (params.scalars.geo)
    {
        resize(&state->pos, size);
        resize(&state->dir, size);
        resize(&state->volume, size);
    }
    if (params.scalars.phys)
    {
        resize(&state->particle, size);
        resize(&state->kinetic_energy, size);
        resize(&state->energy_deposition, size);
    }
    if (params.scalars.post_step)
    {
        resize(&state->step_length, size);
        resize(&state->action, size);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
