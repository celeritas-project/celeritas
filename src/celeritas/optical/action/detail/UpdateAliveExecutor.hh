//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file optical/action/detail/UpdateAliveExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/optical/CoreState.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Update the num_alive counter based on the number of photons that are still
 * alive after compacting vacancies.
 */
struct UpdateAliveExecutor
{
    //// DATA ////

    CRefPtr<CoreParamsData, MemSpace::native> params;
    RefPtr<CoreStateData, MemSpace::native> state;
    size_type state_size;

    //// FUNCTIONS ////

    // Update number of photons that are still alive
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update number of photons that are still alive after compacting vacancies.
 */
CELER_FORCEINLINE_FUNCTION void UpdateAliveExecutor::operator()(ThreadId tid)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);
    CELER_EXPECT(tid.get() == 0);  // Should be called with only one thread

    auto counters = state->init.counters.data().get();
    counters->num_alive = state_size - counters->num_vacancies;
    CELER_ASSERT(state_size >= counters->num_vacancies);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
