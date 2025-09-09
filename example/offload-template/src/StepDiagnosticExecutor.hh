//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/StepDiagnosticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <celeritas/Quantities.hh>
#include <celeritas/global/CoreTrackView.hh>
#include <corecel/Macros.hh>
#include <corecel/Types.hh>

#include "StepDiagnosticData.hh"

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
struct StepDiagnosticExecutor
{
    // Note: this is shorthand for
    // StepParamsData<Ownership::const_reference, MemSpace::native>
    // where native is "host" when running from a .cc file and "device" when
    // running from a .cu file.
    NativeCRef<StepParamsData> const params;
    // Same here but non-const (state data can change)
    NativeRef<StepStateData> const state;

    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Collect statistics about the current step.
 */
CELER_FUNCTION void
StepDiagnosticExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);

    // We're always "event zero" (simultaneous event ID)
    auto& stats = state.data[EventId{0}];

    // Tally the track length
    atomic_add(&stats.step_length, track.sim().step_length());

    // Tally energy deposition over the step
    atomic_add(
        &stats.energy_deposition,
        value_as<units::MevEnergy>(track.physics_step().energy_deposition()));
}

//---------------------------------------------------------------------------//
}  // namespace example
}  // namespace celeritas
