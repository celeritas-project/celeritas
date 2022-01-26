//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ProcessSecondariesLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Atomics.hh"
#include "geometry/GeoTrackView.hh"
#include "sim/TrackData.hh"
#include "sim/TrackInitData.hh"
#include "sim/SimTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondaries.
 */
template<MemSpace M>
class ProcessSecondariesLauncher
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
    CELER_FUNCTION ProcessSecondariesLauncher(const ParamsDataRef& params,
                                              const StateDataRef&  states,
                                              const TrackInitStateDataRef& data)
        : params_(params), states_(states), data_(data)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
        CELER_EXPECT(data_);
    }

    // Create track initializers from secondaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsDataRef&         params_;
    const StateDataRef&          states_;
    const TrackInitStateDataRef& data_;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void
ProcessSecondariesLauncher<M>::operator()(ThreadId tid) const
{
    // Construct the state accessors
    GeoTrackView geo(params_.geometry, states_.geometry, tid);
    SimTrackView sim(states_.sim, tid);

    // Offset in the vector of track initializers
    size_type offset = data_.secondary_counts[tid];

    // A new track was initialized from a secondary in the parent's track slot
    bool initialized = false;

    // Save the parent ID since it will be overwritten if a secondary is
    // initialized in this slot
    const TrackId parent_id(sim.track_id());

    Interaction& result = states_.interactions[tid];
    for (const auto& secondary : result.secondaries)
    {
        if (secondary)
        {
            // Calculate the track ID of the secondary
            // TODO: This is nondeterministic; we need to calculate the
            // track ID in a reproducible way.
            CELER_ASSERT(sim.event_id() < data_.track_counters.size());
            TrackId::size_type track_id = atomic_add(
                &data_.track_counters[sim.event_id()], size_type{1});

            // Create a track initializer from the secondary
            TrackInitializer init;
            init.sim.track_id         = TrackId{track_id};
            init.sim.parent_id        = parent_id;
            init.sim.event_id         = sim.event_id();
            init.sim.alive            = true;
            init.geo.pos              = geo.pos();
            init.geo.dir              = secondary.direction;
            init.particle.particle_id = secondary.particle_id;
            init.particle.energy      = secondary.energy;

            if (!initialized && !sim.alive())
            {
                ParticleTrackView particle(
                    params_.particles, states_.particles, tid);
                PhysicsTrackView phys(
                    params_.physics, states_.physics, {}, {}, tid);

                // The parent was killed, so initialize the first secondary in
                // the parent's track slot. Keep the parent's geometry state
                // but get the direction from the secondary. The material state
                // will be the same as the parent's.
                sim = init.sim;
                geo = GeoTrackView::DetailedInitializer{geo, init.geo.dir};
                particle    = init.particle;
                phys        = {};
                initialized = true;
            }
            else
            {
                // Store the track initializer
                ThreadId init_id(data_.initializers.size()
                                 - data_.parents.size() + offset);
                CELER_ASSERT(init_id < data_.initializers.size());
                data_.initializers[init_id] = init;

                // Store the thread ID of the secondary's parent
                CELER_ASSERT(offset < data_.parents.size());
                data_.parents[ThreadId(offset++)] = tid;
            }
        }
    }
    // Clear the interaction
    result = initialized ? Interaction::from_spawned()
                         : Interaction::from_processed();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
