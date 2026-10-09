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
 *
 * With \c TrackOrder::init_charge, the \c indices array holds the inclusive
 * prefix sum \f$ S \f$ of neutral flags over the \f$ n \f$ initializers used
 * in this step (see \c scan_neutral_initializers). With
 * \f$ N = S_{n-1} \f$ neutral tracks in total, initializer \f$ i \f$ is
 * placed in vacancy \f$ S_i - 1 \f$ if it is neutral and in vacancy
 * \f$ v - (n - N) + (i - S_i) \f$ if it is charged, where \f$ v \f$ is the
 * number of vacancies. Neutral tracks thus fill the front of the vacancies and
 * charged tracks the back, each in initializer order.
 */
CELER_FUNCTION void InitTracksExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    auto const& data = state->init;
    auto* counters = state->init.counters.data().get();
    size_type num_init
        = min(counters->num_vacancies, counters->num_initializers);
    CELER_EXPECT(num_init <= state->size());
    if (tid < num_init)
    {
        // Get the track initializer from the back of the vector. Since new
        // initializers are pushed to the back of the vector, these will be the
        // most recently added and therefore the ones that still might have a
        // parent they can copy the geometry state from.
        TrackInitializer& init = data.initializers[ItemId<TrackInitializer>(
            index_before(counters->num_initializers, tid))];

        // View to the new track to be initialized
        CoreTrackView vacancy{
            *params, *state, [&] {
                if (params->init.track_order == TrackOrder::init_charge)
                {
                    // Index among the initializers used in this step
                    size_type i = index_before(num_init, tid);
                    size_type num_neutral_upto = data.indices[TrackSlotId(i)];
                    if (IsNeutral{params}(init))
                    {
                        // Get the vacancy from the front of the track state
                        CELER_ASSERT(num_neutral_upto > 0);
                        return data.vacancies[TrackSlotId(num_neutral_upto - 1)];
                    }
                    // Get the vacancy from the back of the track state
                    size_type num_charged
                        = num_init - data.indices[TrackSlotId(num_init - 1)];
                    return data.vacancies[TrackSlotId(
                        counters->num_vacancies - num_charged
                        + (i - num_neutral_upto))];
                }
                // Get the vacancy from the back of the track state
                return data.vacancies[TrackSlotId(
                    index_before(counters->num_vacancies, tid))];
            }()};

        vacancy = init;
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
