//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/SetPending.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"

#include "../CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
template<MemSpace M>
class CoreState;

namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Reset the num_pending counter based on the number of primaries.
 */
struct SetPendingExecutor
{
    //// DATA ////

    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    size_type primaries;

    //// FUNCTIONS ////

    // Set num_pending to the number of of primaries waiting to be generated
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid);
};

//---------------------------------------------------------------------------//

void set_pending(CoreParams const& params,
                 CoreState<MemSpace::host>& state,
                 size_type num_primaries);
void set_pending(CoreParams const& params,
                 CoreState<MemSpace::device>& state,
                 size_type num_primaries);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Set num_pending to the number of primaries waiting to be generated.
 */
CELER_FORCEINLINE_FUNCTION void SetPendingExecutor::operator()(ThreadId tid)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(tid.get() == 0);  // Should call with only one thread
    CELER_EXPECT(primaries > 0);

    auto counters = state->init.counters.data().get();
    counters->num_pending = primaries;
    return;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
