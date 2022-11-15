//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/ProcessSecondariesLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/Atomics.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

#include "../SimTrackView.hh"

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
    using ParamsRef = CoreParamsData<Ownership::const_reference, M>;
    using StateRef  = CoreStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION
    ProcessSecondariesLauncher(const CoreRef<M>& core_data)
        : params_(core_data.params), states_(core_data.states)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
    }

    // Create track initializers from secondaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsRef& params_;
    const StateRef&  states_;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondaries.
 */
template<MemSpace M>
CELER_FUNCTION void
ProcessSecondariesLauncher<M>::operator()(ThreadId tid) const
{
    SimTrackView sim(states_.sim, tid);
    if (sim.status() == TrackStatus::inactive)
    {
        // Do not create secondaries from stale data on inactive tracks
        return;
    }

    // Offset in the vector of track initializers
    const auto& data = states_.init;
    CELER_ASSERT(data.secondary_counts[tid] <= data.num_secondaries);
    size_type offset = data.num_secondaries - data.secondary_counts[tid];

    // A new track was initialized from a secondary in the parent's track slot
    bool initialized = false;

    // Save the parent ID since it will be overwritten if a secondary is
    // initialized in this slot
    const TrackId parent_id{sim.track_id()};

    PhysicsStepView phys(params_.physics, states_.physics, tid);
    for (const auto& secondary : phys.secondaries())
    {
        if (secondary)
        {
            // Particles should not be making secondaries while crossing a
            // surface
            GeoTrackView geo(params_.geometry, states_.geometry, tid);
            CELER_ASSERT(!geo.is_on_boundary());

            // Increment the total number of tracks created for this event and
            // calculate the track ID of the secondary
            // TODO: This is nondeterministic; we need to calculate the
            // track ID in a reproducible way.
            CELER_ASSERT(sim.event_id() < data.track_counters.size());
            TrackId::size_type track_id = atomic_add(
                &data.track_counters[sim.event_id()], size_type{1});

            // Create a track initializer from the secondary
            TrackInitializer ti;
            ti.sim.track_id         = TrackId{track_id};
            ti.sim.parent_id        = parent_id;
            ti.sim.event_id         = sim.event_id();
            ti.sim.num_steps        = 0;
            ti.sim.time             = sim.time();
            ti.sim.status           = TrackStatus::alive;
            ti.geo.pos              = geo.pos();
            ti.geo.dir              = secondary.direction;
            ti.particle.particle_id = secondary.particle_id;
            ti.particle.energy      = secondary.energy;

            if (!initialized && sim.status() != TrackStatus::alive)
            {
                ParticleTrackView particle(
                    params_.particles, states_.particles, tid);
                PhysicsTrackView phys(
                    params_.physics, states_.physics, {}, {}, tid);

                // The parent was killed, so initialize the first secondary in
                // the parent's track slot. Keep the parent's geometry state
                // but get the direction from the secondary. Reset the physics
                // state so the multiple scattering range properties are
                // cleared. The material state will be the same as the
                // parent's.
                sim      = ti.sim;
                geo      = GeoTrackView::DetailedInitializer{geo, ti.geo.dir};
                particle = ti.particle;
                phys        = {};
                initialized = true;

                // TODO: make it easier to determine what states need to be
                // reset: the physics MFP, for example, is OK to preserve
            }
            else
            {
                // Store the track initializer
                CELER_ASSERT(offset > 0 && offset <= data.initializers.size());
                data.initializers[ThreadId(data.initializers.size() - offset)]
                    = ti;

                // Store the thread ID of the secondary's parent if the
                // secondary could be initialized in the next step
                if (offset <= data.parents.size())
                {
                    data.parents[ThreadId(data.parents.size() - offset)] = tid;
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
    CELER_ENSURE(sim.status() != TrackStatus::killed);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
