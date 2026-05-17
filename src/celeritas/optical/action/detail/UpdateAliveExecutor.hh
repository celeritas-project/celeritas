//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file optical/action/detail/UpdateAliveExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackView.hh"

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

    size_type state_size;

    //// FUNCTIONS ////

    // Update number of photons that are still alive
    CELER_FORCEINLINE_FUNCTION void operator()(CoreTrackView& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update number of photons that are still alive after compacting vacancies.
 */
CELER_FORCEINLINE_FUNCTION void
UpdateAliveExecutor::operator()(CoreTrackView& track)
{
    CELER_EXPECT(track.thread_id() == ThreadId{0});  // single thread kernel

    track.counters().num_alive = state_size - track.counters().num_vacancies;
    CELER_ASSERT(state_size >= track.counters().num_vacancies);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
