//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/UpdateNewTracksExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

#include "../TrackInitData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Update state counters based on the number of new tracks.
 */
struct UpdateNewTracksExecutor
{
    //// TYPES ////

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    ParamsPtr params;
    StatePtr state;

    //// FUNCTIONS ////

    // Update state counters based on the number of primaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
/*!
 * Update state counters based on the number of new tracks.
 */
CELER_FUNCTION void UpdateNewTracksExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(tid.get() == 0);  // Should call with only one thread

    auto* counters = state->init.counters.data().get();

    size_type num_new_tracks
        = min(counters->num_vacancies, counters->num_initializers);
    if (num_new_tracks > 0)
    {
        // Update initializers/vacancies
        counters->num_initializers -= num_new_tracks;
        counters->num_vacancies -= num_new_tracks;
    }
    // Store number of active tracks at the start of the loop
    counters->num_active = state->size() - counters->num_vacancies;
    return;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
