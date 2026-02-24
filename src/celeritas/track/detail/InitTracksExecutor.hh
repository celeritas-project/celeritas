//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/InitTracksExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

#include "../CoreStateCounters.hh"
#include "../SimTrackView.hh"
#include "../Utils.hh"

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
    size_type num_init{};

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
    CELER_EXPECT(tid < num_init);

    auto const& data = state->init;
    auto counters = state->init.counters.data().get();
    // Get the track initializer from the back of the vector. Since new
    // initializers are pushed to the back of the vector, these will be the
    // most recently added and therefore the ones that still might have a
    // parent they can copy the geometry state from.
    TrackInitializer& init = data.initializers[ItemId<TrackInitializer>([&] {
        if (params->init.track_order == TrackOrder::init_charge)
        {
            // Get the index into the track initializer or parent track slot ID
            // array from the sorted indices
            return data.indices[TrackSlotId(index_before(num_init, tid))]
                   + counters->num_initializers - num_init;
        }
        return index_before(counters->num_initializers, tid);
    }())];

    // View to the new track to be initialized
    CoreTrackView vacancy{
        *params, *state, [&] {
            if (params->init.track_order == TrackOrder::init_charge
                && IsNeutral{params}(init))
            {
                // Get the vacancy from the front of the track state
                return data.vacancies[TrackSlotId(index_before(num_init, tid))];
            }
            // Get the vacancy from the back of the track state
            return data.vacancies[TrackSlotId(
                index_before(counters->num_vacancies, tid))];
        }()};

    // Clear parent IDs if new primaries were added this step
    if (counters->num_generated)
    {
        init.geo.parent = {};
    }

    vacancy = init;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
