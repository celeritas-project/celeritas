//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadPreGatherExecutor.hh
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
 * Gather pre-step data needed to generate optical distributions.
 */
struct OffloadPreGatherExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeRef<OffloadPreStateData> const state;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data needed to generate optical distributions.
 */
CELER_FUNCTION void
OffloadPreGatherExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);
    CELER_EXPECT(track.track_slot_id() < state.step.size());

    OffloadPreStepData& step = state.step[track.track_slot_id()];
    step.material = track.material().material_record().optical_material_id();
    if (step.material)
    {
        step.speed = track.particle().speed();
        step.pos = track.geometry().pos();
        step.time = track.sim().time();
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
