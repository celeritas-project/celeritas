//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/InitTracksExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

#include "Utils.hh"
#include "../CoreStateCounters.hh"
#include "../SimTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/cont/ArrayIO.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 *
 * The track initializers are created from either primary particles or
 * secondaries. The new tracks are inserted into empty slots (vacancies) in the
 * track vector.
 */
struct InitTracksExecutor
{
    //// TYPES ////

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    ParamsPtr params;
    StatePtr state;
    size_type num_new_tracks;
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Initialize track states
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 *
 * The track initializers are created from either primary particles or
 * secondaries. The new tracks are inserted into empty slots (vacancies) in the
 * track vector.
 */
CELER_FUNCTION void InitTracksExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(tid < num_new_tracks);

    // Get the track initializer from the back of the vector. Since new
    // initializers are pushed to the back of the vector, these will be the
    // most recently added and therefore the ones that still might have a
    // parent they can copy the geometry state from.
    auto const& data = state->init;
    ItemId<TrackInitializer> idx{index_before(counters.num_initializers, tid)};
    TrackInitializer const& init = data.initializers[idx];

    // Thread ID of vacant track where the new track will be initialized
    TrackSlotId vacancy = [&] {
        TrackSlotId idx{index_before(counters.num_vacancies, tid)};
        return data.vacancies[idx];
    }();

    // Initialize the simulation state
    {
        SimTrackView sim(params->sim, state->sim, vacancy);
        sim = init.sim;
    }

    // Initialize the particle physics data
    {
        ParticleTrackView particle(
            params->particles, state->particles, vacancy);
        particle = init.particle;
    }

    // Initialize the geometry
    {
        GeoTrackView geo(params->geometry, state->geometry, vacancy);
        if (tid < counters.num_secondaries)
        {
            // Copy the geometry state from the parent for improved
            // performance
            TrackSlotId parent_id = data.parents[TrackSlotId{
                index_before(data.parents.size(), tid)}];
            GeoTrackView const parent_geo(
                params->geometry, state->geometry, parent_id);
            geo = GeoTrackView::DetailedInitializer{parent_geo, init.geo.dir};
        }
        else
        {
            // Initialize it from the position (more expensive)
            geo = init.geo;
#if !CELER_DEVICE_COMPILE
            // TODO: kill particle with 'error' state if this happens
            CELER_VALIDATE(!geo.is_outside(),
                           << "track started outside the geometry at "
                           << init.geo.pos);
#endif
        }

        // Initialize the material
        auto matid
            = GeoMaterialView(params->geo_mats).material_id(geo.volume_id());
#if !CELER_DEVICE_COMPILE
        // TODO: kill particle with 'error' state if this happens
        CELER_VALIDATE(matid,
                       << "track started in an unknown material (volume ID "
                       << geo.volume_id().unchecked_get() << ")");
#endif
        MaterialTrackView mat(params->materials, state->materials, vacancy);
        mat = {matid};
    }

    // Initialize the physics state
    {
        PhysicsTrackView phys(params->physics, state->physics, {}, {}, vacancy);
        phys = {};
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
