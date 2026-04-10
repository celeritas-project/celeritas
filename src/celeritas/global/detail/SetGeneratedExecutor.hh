//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/SetGeneratedExecutor.hh
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
 *     // Initialize the num_generated counter to zero.
 */
struct SetGeneratedExecutor
{
    //// DATA ////

    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;

    //// FUNCTIONS ////

    // Initialize the num_generated counter to zero
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid);
};

//---------------------------------------------------------------------------//

void set_generated(CoreParams const& params, CoreState<MemSpace::host>& state);
void set_generated(CoreParams const& params,
                   CoreState<MemSpace::device>& state);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Initialize the num_generated counter to zero.
 */
CELER_FORCEINLINE_FUNCTION void SetGeneratedExecutor::operator()(ThreadId tid)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(tid.get() == 0);  // Should call with only one thread

    auto* counters = state->init.counters.data().get();
    counters->num_generated = 0;
    return;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
