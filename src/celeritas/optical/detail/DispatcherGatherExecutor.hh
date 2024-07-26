//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/DispatcherGatherExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/DispatcherData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
struct DispatcherGatherExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeRef<DispatcherStateData> const state;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data needed to generate optical distributions.
 */
CELER_FUNCTION void
DispatcherGatherExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);
    CELER_EXPECT(track.track_slot_id() < state.step.size());

    DispatcherPreStepData& step = state.step[track.track_slot_id()];
    step.speed = track.make_particle_view().speed();
    step.pos = track.make_geo_view().pos();
    step.time = track.make_sim_view().time();
    step.opt_mat
        = track.make_material_view().make_material_view().optical_material_id();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
