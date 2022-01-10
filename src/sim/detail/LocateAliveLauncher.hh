//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LocateAliveLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Atomics.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "sim/TrackData.hh"
#include "sim/TrackInitData.hh"
#include "sim/SimTrackView.hh"
#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Determine which tracks are alive and count secondaries.
 *
 * This finds empty slots in the track vector and counts the number of
 * secondaries created in each interaction. If the track was killed and
 * produced secondaries, the empty track slot is filled with the first
 * secondary.
 */
template<MemSpace M>
class LocateAliveLauncher
{
  public:
    //!@{
    //! Type aliases
    using ParamsDataRef         = ParamsData<Ownership::const_reference, M>;
    using StateDataRef          = StateData<Ownership::reference, M>;
    using TrackInitStateDataRef = TrackInitStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION LocateAliveLauncher(const ParamsDataRef&         params,
                                       const StateDataRef&          states,
                                       const TrackInitStateDataRef& data)
        : params_(params), states_(states), data_(data)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
        CELER_EXPECT(data_);
    }

    // Determine which tracks are alive and count secondaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsDataRef&         params_;
    const StateDataRef&          states_;
    const TrackInitStateDataRef& data_;
};

//---------------------------------------------------------------------------//
/*!
 * Determine which tracks are alive and count secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void LocateAliveLauncher<M>::operator()(ThreadId tid) const
{
    // Index of the secondary to copy to the parent track if vacant
    size_type secondary_idx = flag_id();

    // Count how many secondaries survived cutoffs for each track
    data_.secondary_counts[tid] = 0;
    Interaction& result         = states_.interactions[tid];
    for (auto i : range(result.secondaries.size()))
    {
        if (result.secondaries[i])
        {
            if (secondary_idx == flag_id())
            {
                secondary_idx = i;
            }
            ++data_.secondary_counts[tid];
        }
    }

    SimTrackView sim(states_.sim, tid);
    if (sim.alive())
    {
        // The track is alive: mark this track slot as occupied
        data_.vacancies[tid] = flag_id();
    }
    else if (secondary_idx != flag_id())
    {
        // The track was killed and it produced secondaries: fill the empty
        // track slot with the first secondary and mark as occupied

        // Calculate the track ID of the secondary
        // TODO: This is nondeterministic; we need to calculate the track
        // ID in a reproducible way.
        CELER_ASSERT(sim.event_id() < data_.track_counters.size());
        TrackId::size_type track_id
            = atomic_add(&data_.track_counters[sim.event_id()], size_type{1});

        // Initialize the simulation state
        sim = {TrackId{track_id}, sim.track_id(), sim.event_id(), true};

        // Initialize the particle state from the secondary
        Secondary&        secondary = result.secondaries[secondary_idx];
        ParticleTrackView particle(params_.particles, states_.particles, tid);
        particle = {secondary.particle_id, secondary.energy};

        // Keep the parent's geometry state but get the direction from the
        // secondary. The material state will be the same as the parent's.
        GeoTrackView geo(params_.geometry, states_.geometry, tid);
        geo = GeoTrackView::DetailedInitializer{geo, secondary.direction};

        // Initialize the physics state
        PhysicsTrackView phys(params_.physics, states_.physics, {}, {}, tid);
        phys = {};

        // Mark the secondary as processed and the track as active
        --data_.secondary_counts[tid];
        secondary            = Secondary{};
        data_.vacancies[tid] = flag_id();
    }
    else
    {
        // The track was killed and did not produce secondaries: store the
        // index so it can be used later to initialize a new track
        data_.vacancies[tid] = tid.get();
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
