//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/ProcessSecondariesLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Atomics.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/phys/Secondary.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/TrackInitData.hh"

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
    //! \name Type aliases
    using ParamsRef = CoreParamsData<Ownership::const_reference, M>;
    using StateRef = CoreStateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION
    ProcessSecondariesLauncher(ParamsRef const& params, StateRef const& states)
        : params_(params), states_(states)
    {
        CELER_EXPECT(params_);
        CELER_EXPECT(states_);
    }

    // Create track initializers from secondaries
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        // The grid size should be equal to the state size and no thread/slot
        // remapping should be performed
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }

  private:
    ParamsRef const& params_;
    StateRef const& states_;
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondaries.
 *
 * This kernel is launched with a grid size equal to the number of track slots,
 * so ThreadId should be equal to TrackSlotId. No remapping should be done.
 */
template<MemSpace M>
CELER_FUNCTION void
ProcessSecondariesLauncher<M>::operator()(TrackSlotId tid) const
{
    SimTrackView sim(params_.sim, states_.sim, tid);
    if (sim.status() == TrackStatus::inactive)
    {
        // Do not create secondaries from stale data on inactive tracks
        return;
    }

    // Offset in the vector of track initializers
    auto const& data = states_.init;
    CELER_ASSERT(data.secondary_counts[tid] <= data.scalars.num_secondaries);
    size_type offset = data.scalars.num_secondaries
                       - data.secondary_counts[tid];

    // A new track was initialized from a secondary in the parent's track slot
    bool initialized = false;

    // Save the parent ID since it will be overwritten if a secondary is
    // initialized in this slot
    const TrackId parent_id{sim.track_id()};

    PhysicsStepView phys(params_.physics, states_.physics, tid);
    for (auto const& secondary : phys.secondaries())
    {
        if (secondary)
        {
            CELER_ASSERT(secondary.energy > zero_quantity()
                         && is_soft_unit_vector(secondary.direction));

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
            ti.sim.track_id = TrackId{track_id};
            ti.sim.parent_id = parent_id;
            ti.sim.event_id = sim.event_id();
            ti.sim.time = sim.time();
            ti.sim.status = TrackStatus::alive;
            ti.geo.pos = geo.pos();
            ti.geo.dir = secondary.direction;
            ti.particle.particle_id = secondary.particle_id;
            ti.particle.energy = secondary.energy;
            CELER_ASSERT(ti);

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
                sim = ti.sim;
                geo = GeoTrackView::DetailedInitializer{geo, ti.geo.dir};
                particle = ti.particle;
                phys = {};
                initialized = true;

                // TODO: make it easier to determine what states need to be
                // reset: the physics MFP, for example, is OK to preserve
            }
            else
            {
                // Store the track initializer
                CELER_ASSERT(offset > 0
                             && offset <= data.scalars.num_initializers);
                data.initializers[ItemId<TrackInitializer>{
                    data.scalars.num_initializers - offset}]
                    = ti;

                // Store the thread ID of the secondary's parent if the
                // secondary could be initialized in the next step
                if (offset <= data.parents.size())
                {
                    data.parents[TrackSlotId(data.parents.size() - offset)]
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
    CELER_ENSURE(sim.status() != TrackStatus::killed);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
