//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/UpdatePendingExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/track/Utils.hh"

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
 * Update the num_pending counter based on the generated photons from buffered
 * optical distribution data.
 */
struct UpdatePendingExecutor
{
    //// DATA ////

    RefPtr<CoreStateData, MemSpace::native> state;
    size_type num_photons;

    //// FUNCTIONS ////

    // Update number of of primaries waiting to be generated
    CELER_FORCEINLINE_FUNCTION void operator()(ThreadId tid);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update number of primaries to be generated to include the buffered optical
 * photons.
 */
CELER_FORCEINLINE_FUNCTION void UpdatePendingExecutor::operator()(ThreadId tid)
{
    CELER_EXPECT(state);
    CELER_EXPECT(tid.get() == 0);  // Should call with only one thread

    auto counters = state->init.counters.data().get();
    counters->num_pending += num_photons;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
