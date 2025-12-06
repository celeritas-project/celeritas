//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadPrePostGatherExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../OffloadData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Gather state data needed to generate optical distributions.
 */
struct OffloadPrePostGatherExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeRef<OffloadPrePostStateData> const state;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Gather post-along-step data needed for scintillation optical distributions.
 */
CELER_FUNCTION void
OffloadPrePostGatherExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);
    CELER_EXPECT(track.track_slot_id() < state.step.size());

    OffloadPrePostStepData& step = state.step[track.track_slot_id()];
    step.speed = track.particle().speed();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
