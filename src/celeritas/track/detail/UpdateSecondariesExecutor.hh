//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/UpdateSecondariesExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Copier.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Update the num_secondaries and num_initializers and then check if there is
 * room for all the secondaries. If so, update the number of alive tracks to
 * include these secondaries.
 */
struct UpdateSecondariesExecutor
{
    //// DATA ////
    RefPtr<CoreStateData, MemSpace::native> state;

    //// FUNCTIONS ////

    // Update the secondaries and initializers and alive tracks if there is
    // space for all the secondaries
    CELER_FORCEINLINE_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update the num_secondaries and num_initializers and then check if there is
 * room for all the secondaries. If so, update the number of alive tracks to
 * include these secondaries.
 */
CELER_FORCEINLINE_FUNCTION void UpdateSecondariesExecutor::operator()(
    CoreTrackView& track)
{
    CELER_EXPECT(track.thread_id() == ThreadId{0});  // single thread kernel

    // The last entry in the secondary_counts array holds the exclusive sum,
    // which is the number of secondaries
    auto const* data = state->init.secondary_counts.data().get();
    auto const* end = data + state->init.secondary_counts.size() - 1;
    // Update the number of secondaries and number of initializers based on
    // these new secondaries
    track.counters().num_secondaries = *end;
    track.counters().num_initializers += *end;
    track.counters().num_alive = state->size() - track.counters().num_vacancies;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
