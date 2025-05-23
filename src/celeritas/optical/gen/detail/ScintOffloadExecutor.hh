//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/ScintOffloadExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../OffloadData.hh"
#include "../ScintillationOffload.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 *
 * Note that the track may be inactive! TODO: we could add a `user_start`
 * action to clear distribution data rather than applying it to inactive tracks
 * at every step.
 */
struct ScintOffloadExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<ScintillationData> const scintillation;
    NativeRef<OffloadStateData> const state;
    OpticalOffloadCounters<> size;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
CELER_FUNCTION void ScintOffloadExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);

    using DistId = ItemId<GeneratorDistributionData>;

    auto tsid = track.track_slot_id();
    CELER_ASSERT(size.scintillation + tsid.get() < state.scintillation.size());
    auto& scintillation_dist
        = state.scintillation[DistId(size.scintillation + tsid.get())];

    // Clear distribution data
    scintillation_dist = {};

    auto sim = track.sim();
    auto const& step = state.step[tsid];

    if (!step || sim.status() == TrackStatus::inactive)
    {
        // Inactive tracks, materials with no optical properties, or particles
        // that started the step with zero energy (e.g. a stopped positron)
        return;
    }

    Real3 const& pos = track.geometry().pos();
    auto edep = track.physics_step().energy_deposition();
    auto particle = track.particle();
    auto rng = track.rng();

    // Get the distribution data used to generate scintillation optical photons
    ScintillationOffload generate(
        particle, sim, pos, edep, scintillation, step);
    scintillation_dist = generate(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
