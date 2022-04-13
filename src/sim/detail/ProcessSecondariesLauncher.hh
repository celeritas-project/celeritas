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
#include "physics/base/PhysicsTrackView.hh"
#include "sim/CoreTrackData.hh"
#include "sim/SimTrackView.hh"
#include "sim/TrackInitData.hh"

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
    using ParamsRef         = CoreParamsData<Ownership::const_reference, M>;
    using StateRef          = CoreStateData<Ownership::reference, M>;
    using TrackInitStateRef = TrackInitStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION
    ProcessSecondariesLauncher(const CoreRef<M>&        core_data,
                               const TrackInitStateRef& init_data)
        : params_(core_data.params), states_(core_data.states), data_(init_data)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
        CELER_EXPECT(data_);
    }

    // Create track initializers from secondaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsRef&         params_;
    const StateRef&          states_;
    const TrackInitStateRef& data_;
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
    GeoTrackView     geo(params_.geometry, states_.geometry, tid);
    SimTrackView     sim(states_.sim, tid);
    PhysicsTrackView phys(params_.physics, states_.physics, {}, {}, tid);

    // Offset in the vector of track initializers
    CELER_ASSERT(data_.secondary_counts[tid] <= data_.num_secondaries);
    size_type offset = data_.num_secondaries - data_.secondary_counts[tid];

    // A new track was initialized from a secondary in the parent's track slot
    bool initialized = false;

    // Save the parent ID since it will be overwritten if a secondary is
    // initialized in this slot
    const TrackId parent_id{sim.track_id()};

    for (const auto& secondary : phys.secondaries())
    {
        if (secondary)
        {
            // Particles should not be making secondaries while crossing a
            // surface
            CELER_ASSERT(!geo.surface_id());

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
            init.sim.num_steps        = 0;
            init.sim.status           = TrackStatus::alive;
            init.geo.pos              = geo.pos();
            init.geo.dir              = secondary.direction;
            init.particle.particle_id = secondary.particle_id;
            init.particle.energy      = secondary.energy;

            if (!initialized && sim.status() != TrackStatus::alive)
            {
                ParticleTrackView particle(
                    params_.particles, states_.particles, tid);

                // The parent was killed, so initialize the first secondary in
                // the parent's track slot. Keep the parent's geometry state
                // but get the direction from the secondary. The material state
                // will be the same as the parent's.
                sim = init.sim;
                geo = GeoTrackView::DetailedInitializer{geo, init.geo.dir};
                particle    = init.particle;
                initialized = true;
            }
            else
            {
                // Store the track initializer
                CELER_ASSERT(offset > 0 && offset <= data_.initializers.size());
                data_.initializers[ThreadId(data_.initializers.size() - offset)]
                    = init;

                // Store the thread ID of the secondary's parent if the
                // secondary could be initialized in the next step
                if (offset <= data_.parents.size())
                {
                    data_.parents[ThreadId(data_.parents.size() - offset)]
                        = tid;
                }
                --offset;
            }
        }
    }

    if (!initialized && sim.status() == TrackStatus::killed)
    {
        // Track is no longer used as part of transport
        sim.status(TrackStatus::inactive);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
