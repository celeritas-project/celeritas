//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/detail/StepDiagnosticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/user/ParticleTallyData.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
struct StepDiagnosticExecutor
{
    NativeCRef<ParticleTallyParamsData> const params;
    NativeRef<ParticleTallyStateData> const state;

    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Collect distribution of steps per track for each particle type.
 */
CELER_FUNCTION void
StepDiagnosticExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    // Tally the number of steps if the track was killed
    auto sim = track.sim();
    if (sim.status() == TrackStatus::killed)
    {
        size_type index = celeritas::min(sim.num_steps(), params.num_bins - 1);
        using namespace celeritas::literals;
        atomic_add(&state.counts[ItemId<size_type>(index)], 1_sz);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
