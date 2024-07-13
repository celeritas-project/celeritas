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
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

#include "Utils.hh"
#include "../CoreStateCounters.hh"
#include "../SimTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
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
    size_type partition_index;
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
    CELER_EXPECT(partition_index <= num_new_tracks);

    // Get the track initializer from the back of the vector. Since new
    // initializers are pushed to the back of the vector, these will be the
    // most recently added and therefore the ones that still might have a
    // parent they can copy the geometry state from.
    auto const& data = state->init;
    ItemId<TrackInitializer> idx{index_before(counters.num_initializers, tid)};

    // View to the new track to be initialized
    CoreTrackView vacancy{
        *params, *state, [&] {
            TrackSlotId idx{index_partitioned(num_new_tracks,
                                              counters.num_vacancies,
                                              num_new_tracks - partition_index,
                                              tid)};
            return data.vacancies[idx];
        }()};

    // Initialize the simulation state and particle attributes
    TrackInitializer const& init = data.initializers[idx];
    vacancy.make_sim_view() = init.sim;
    vacancy.make_particle_view() = init.particle;

    // Initialize the geometry
    {
        auto geo = vacancy.make_geo_view();
        if (tid < counters.num_secondaries)
        {
            // Copy the geometry state from the parent for improved performance
            TrackSlotId parent_id = data.parents[TrackSlotId{
                index_before(data.parents.size(), tid)}];
            GeoTrackView const parent_geo(
                params->geometry, state->geometry, parent_id);
            geo = GeoTrackView::DetailedInitializer{parent_geo, init.geo.dir};
            CELER_ASSERT(!geo.is_outside());
        }
        else
        {
            // Initialize it from the position (more expensive)
            geo = init.geo;
            if (CELER_UNLIKELY(geo.failed() || geo.is_outside()))
            {
#if !CELER_DEVICE_COMPILE
                if (!geo.failed())
                {
                    // Print an error message if initialization was
                    // "successful" but track is outside
                    CELER_LOG_LOCAL(error) << "Track started outside the "
                                              "geometry";
                }
                else
                {
                    // Do not print anything: the geometry track view itself
                    // should've printed a detailed error message
                }
#endif
                vacancy.apply_errored();
                return;
            }
        }

        // Initialize the material
        auto matid
            = vacancy.make_geo_material_view().material_id(geo.volume_id());
        if (CELER_UNLIKELY(!matid))
        {
#if !CELER_DEVICE_COMPILE
            CELER_LOG_LOCAL(error) << "Track started in an unknown material";
#endif
            vacancy.apply_errored();
            return;
        }
        vacancy.make_material_view() = {matid};
    }

    // Initialize the physics state
    vacancy.make_physics_view() = {};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
