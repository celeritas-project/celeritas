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
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/Secondary.hh"

#include "../CoreStateCounters.hh"

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

    CoreTrackView track(*params, *state, tid);

    auto sim = track.sim();
    if (sim.status() == TrackStatus::inactive)
    {
        // Do not create secondaries from stale data on inactive tracks
        return;
    }

    // Offset in the vector of track initializers
    auto& data = state->init;
    CELER_ASSERT(data.secondary_counts[tid] <= counters.num_secondaries);
    size_type offset = counters.num_secondaries - data.secondary_counts[tid];

    // Save the parent ID since it will be overwritten if a secondary is
    // initialized in this slot
    TrackId const track_id{sim.track_id()};

    for (auto const& secondary : track.physics_step().secondaries())
    {
        if (secondary)
        {
            CELER_ASSERT(secondary.energy > zero_quantity()
                         && is_soft_unit_vector(secondary.direction));

            // Particles should not be making secondaries while crossing a
            // surface
            auto geo = track.geometry();
            CELER_ASSERT(!geo.is_on_boundary());

            // Create a track initializer from the secondary
            TrackInitializer ti;
            ti.sim.track_id = make_track_id(params->init, data, sim.event_id());
            ti.sim.primary_id = sim.primary_id();
            ti.sim.parent_id = track_id;
            ti.sim.event_id = sim.event_id();
            ti.sim.time = sim.time();
            ti.geo.pos = geo.pos();
            ti.geo.dir = secondary.direction;
            ti.particle.particle_id = secondary.particle_id;
            ti.particle.energy = secondary.energy;
            CELER_ASSERT(ti);

            if (sim.track_id() == track_id && sim.status() != TrackStatus::alive
                && params->init.track_order != TrackOrder::init_charge)
            {
                /*!
                 * Skip in-place initialization when tracks are partitioned by
                 * charge to reduce the amount of mixing
                 *
                 * \todo Consider allowing this if the parent's charge is the
                 * same as the secondary's
                 */

                // The parent was killed, so initialize the first secondary in
                // the parent's track slot. Keep the parent's geometry state
                // but get the direction from the secondary.
                ti.geo.parent = tid;
                track = ti;
            }
            else
            {
                CELER_ASSERT(offset > 0 && offset <= counters.num_initializers);

                if (offset <= min(counters.num_secondaries,
                                  counters.num_vacancies)
                    && (params->init.track_order != TrackOrder::init_charge
                        || sim.status() == TrackStatus::alive))
                {
                    // Store the thread ID of the secondary's parent if the
                    // secondary will be initialized in the next step. If the
                    // initializers are partitioned by charge, the in-place
                    // initialization of the secondary is skipped, so another
                    // track might overwrite this state during initialization
                    // unless the track is alive.
                    ti.geo.parent = tid;
                }

                // Store the track initializer
                data.initializers[ItemId<TrackInitializer>{
                    counters.num_initializers - offset}]
                    = ti;

                --offset;
            }
        }
    }

    if (sim.track_id() == track_id && sim.status() == TrackStatus::killed)
    {
        // Track is no longer used as part of transport
        sim.status(TrackStatus::inactive);
    }
    CELER_ENSURE(sim.status() != TrackStatus::killed);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
