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
    //// TYPES ////

    using ParamsPtr = CRefPtr<CoreParamsData, MemSpace::native>;
    using StatePtr = RefPtr<CoreStateData, MemSpace::native>;

    //// DATA ////

    ParamsPtr params;
    StatePtr state;

    size_type num_primaries;

    //// FUNCTIONS ////

    // Update state counters based on the number of primaries
    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
/*!
 * Update state counters based on the number of primaries.
 */
CELER_FUNCTION void UpdateCountersExecutor::operator()(ThreadId tid) const
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(tid.get() == 0);  // Should call with only one thread

    auto* counters = state->init.counters.data().get();
    // Update track initializers from primaries
    counters->num_initializers += num_primaries;
    // Mark that the primaries have been processed
    counters->num_generated += num_primaries;
    counters->num_pending = 0;
    return;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
