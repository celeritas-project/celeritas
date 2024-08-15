//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OffloadGatherExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/OffloadData.hh"

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
    step.speed = track.make_particle_view().speed();
    step.pos = track.make_geo_view().pos();
    step.time = track.make_sim_view().time();
    step.opt_mat
        = track.make_material_view().make_material_view().optical_material_id();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
