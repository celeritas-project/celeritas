//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/ProcessSecondariesExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
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

#include "../CoreStateCounters.hh"
#include "../SimTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondaries.
 */
struct ProcessSecondariesExecutor
{
    //// TYPES ////

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    ParamsPtr params;
    StatePtr state;
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Determine which tracks are alive and count secondaries
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        // The grid size should be equal to the state size and no thread/slot
        // remapping should be performed
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
/*!
 * Create track initializers from secondaries.
 *
 * This kernel is executed with a grid size equal to the number of track
 * slots, so ThreadId should be equal to TrackSlotId. No remapping should be
 * done.
 */
CELER_FUNCTION void
ProcessSecondariesExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(tid < state->size());

    SimTrackView sim(params->sim, state->sim, tid);
    if (sim.status() == TrackStatus::inactive)
    {
        // Do not create secondaries from stale data on inactive tracks
        return;
    }

    // Offset in the vector of track initializers
    auto& data = state->init;
    CELER_ASSERT(data.secondary_counts[tid] <= counters.num_secondaries);
    size_type offset = counters.num_secondaries - data.secondary_counts[tid];

    // A new track was initialized from a secondary in the parent's track slot
    bool initialized = false;

    // Save the parent ID since it will be overwritten if a secondary is
    // initialized in this slot
    TrackId const parent_id{sim.track_id()};

    PhysicsStepView const phys_step(params->physics, state->physics, tid);
    for (auto const& secondary : phys_step.secondaries())
    {
        if (secondary)
        {
            CELER_ASSERT(secondary.energy > zero_quantity()
                         && is_soft_unit_vector(secondary.direction));

            // Particles should not be making secondaries while crossing a
            // surface
            GeoTrackView geo(params->geometry, state->geometry, tid);
            CELER_ASSERT(!geo.is_on_boundary());

            // Create a track initializer from the secondary
            TrackInitializer ti;
            ti.sim.track_id = make_track_id(params->init, data, sim.event_id());
            ti.sim.parent_id = parent_id;
            ti.sim.event_id = sim.event_id();
            ti.sim.time = sim.time();
            ti.geo.pos = geo.pos();
            ti.geo.dir = secondary.direction;
            ti.particle.particle_id = secondary.particle_id;
            ti.particle.energy = secondary.energy;
            CELER_ASSERT(ti);

            if (!initialized && sim.status() != TrackStatus::alive
                && params->init.track_order != TrackOrder::init_charge)
            {
                /*!
                 * Skip in-place initialization when tracks are partitioned by
                 * charge to reduce the amount of mixing
                 *
                 * \todo Consider allowing this if the parent's charge is the
                 * same as the secondary's
                 */

                ParticleTrackView particle(
                    params->particles, state->particles, tid);
                PhysicsTrackView phys(
                    params->physics, state->physics, particle, {}, tid);

                // The parent was killed, so initialize the first secondary in
                // the parent's track slot. Keep the parent's geometry state
                // but get the direction from the secondary. Reset the physics
                // state so the multiple scattering range properties are
                // cleared. The material state will be the same as the
                // parent's.
                sim = ti.sim;
                geo = {ti.geo.pos, ti.geo.dir, tid};
                particle = ti.particle;
                phys = {};
                initialized = true;

                /*!
                 * \todo make it easier to determine what states need to be
                 * reset: the physics MFP, for example, is OK to preserve
                 */
            }
            else
            {
                // Store the track initializer
                CELER_ASSERT(offset > 0 && offset <= counters.num_initializers);
                data.initializers[ItemId<TrackInitializer>{
                    counters.num_initializers - offset}]
                    = ti;

                if (offset <= state->size()
                    && (params->init.track_order != TrackOrder::init_charge
                        || sim.status() == TrackStatus::alive))
                {
                    // Store the thread ID of the secondary's parent if the
                    // secondary could be initialized in the next step. If the
                    // tracks are partitioned by charge we skip in-place
                    // initialization of the secondary, so the parent track
                    // must still be alive to ensure the state isn't
                    // overwritten
                    ti.geo.parent = tid;
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
