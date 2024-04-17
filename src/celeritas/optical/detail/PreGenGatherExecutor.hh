//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenGatherExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/OpticalGenData.hh"

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
struct PreGenGatherExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeRef<OpticalGenStateData> const state;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Gather pre-step data needed to generate optical distributions.
 */
CELER_FUNCTION void PreGenGatherExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);

    auto particle = track.make_particle_view();
    if (particle.is_stopped())
    {
        return;
    }

    CELER_ASSERT(track.track_slot_id() < state.step.size());
    OpticalPreStepData& step = state.step[track.track_slot_id()];
    step.speed = particle.speed();
    step.pos = track.make_geo_view().pos();
    step.time = track.make_sim_view().time();

    CELER_ENSURE(step);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
