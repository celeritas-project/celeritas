//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/UpdateNumActiveExecutor.hh
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
 * Update num_active state counter based on the number of vacancies.
 */
struct UpdateNumActiveExecutor
{
    //// DATA ////

    size_type state_size;

    //// FUNCTIONS ////

    // Update state counters based on the number of primaries
    CELER_FORCEINLINE_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
/*!
 * Update number of active trackes based on the number of vacancies.
 */
CELER_FORCEINLINE_FUNCTION void
UpdateNumActiveExecutor::operator()(CoreTrackView& track)
{
    CELER_EXPECT(track.thread_id() == ThreadId{0});  // single thread kernel

    size_type num_new_tracks = min(track.counters().num_vacancies,
                                   track.counters().num_initializers);
    if (num_new_tracks > 0)
    {
        // Update initializers/vacancies
        track.counters().num_initializers -= num_new_tracks;
        track.counters().num_vacancies -= num_new_tracks;
    }
    // Store number of active tracks at the start of the loop
    track.counters().num_active = state_size - track.counters().num_vacancies;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
