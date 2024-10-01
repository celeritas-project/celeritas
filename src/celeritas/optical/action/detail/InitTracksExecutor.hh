//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/InitTracksExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/track/CoreStateCounters.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 *
 * The track initializers are created from either primary particles or
 * secondaries. The new tracks are inserted into the vacancies in the track
 * vector.
 */
struct InitTracksExecutor
{
    //// TYPES ////

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    ParamsPtr params;
    StatePtr state;
    CoreStateCounters counters;

    //// FUNCTIONS ////

    // Initialize track states
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;

    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid) const
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
};

//---------------------------------------------------------------------------//
/*!
 * Initialize the track states.
 */
CELER_FUNCTION void InitTracksExecutor::operator()(TrackSlotId tid) const
{
    CELER_EXPECT(tid < counters.num_initializers);
    CELER_EXPECT(tid < counters.num_vacancies);

    // Create the view to the new track to be initialized
    CoreTrackView vacancy{*params, *state, state->init.vacancies[tid]};

    // Get the initializer from the back of the vector and initialize the track
    vacancy = state->init.initializers[ItemId<TrackInitializer>(
        counters.num_initializers - tid.get() - 1)];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
