//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadGatherExecutor.hh
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
 *
 * TODO: check optical material first, skip storing if it's false. Also maybe
 * skip storing for tracks that can't lose energy over the step?
 */
struct OffloadGatherExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeRef<OffloadStateData> const state;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION void
OffloadGatherExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);
    CELER_EXPECT(track.track_slot_id() < state.step.size());

    OffloadPreStepData& step = state.step[track.track_slot_id()];
    step.speed = track.particle().speed();
    step.pos = track.geometry().pos();
    step.time = track.sim().time();
    step.material = track.material().material_record().optical_material_id();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
