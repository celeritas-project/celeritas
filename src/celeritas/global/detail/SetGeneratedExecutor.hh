//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/SetGeneratedExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackView.hh"

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
 * Initialize the num_generated counter to zero.
 */
struct SetGeneratedExecutor
{
    //// FUNCTIONS ////

    // Initialize the num_generated counter to zero
    CELER_FORCEINLINE_FUNCTION void operator()(CoreTrackView& track);
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
CELER_FORCEINLINE_FUNCTION void
SetGeneratedExecutor::operator()(CoreTrackView& track)
{
    CELER_EXPECT(track.thread_id() == ThreadId{0});  // single thread kernel
    track.counters().num_generated = 0;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
