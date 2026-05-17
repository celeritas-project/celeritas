//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/UpdateCountersExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../TrackInitData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Create track initializers from primary particles.
 */
struct UpdateCountersExecutor
{
    //// DATA ////

    size_type num_primaries;

    //// FUNCTIONS ////

    // Update state counters based on the number of primaries
    CELER_FORCEINLINE_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
/*!
 * Update state counters based on the number of primaries.
 */
CELER_FORCEINLINE_FUNCTION void
UpdateCountersExecutor::operator()(CoreTrackView& track)
{
    CELER_EXPECT(track.thread_id() == ThreadId{0});  // single thread kernel

    // Update track initializers from primaries
    track.counters().num_initializers += num_primaries;
    // Mark that the primaries have been processed
    track.counters().num_generated += num_primaries;
    track.counters().num_pending = 0;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
