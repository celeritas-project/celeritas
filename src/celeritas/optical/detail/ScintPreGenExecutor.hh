//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/ScintPreGenExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/PreGenData.hh"
#include "celeritas/optical/ScintillationPreGenerator.hh"

namespace celeritas
{
namespace optical
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
struct ScintPreGenExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<ScintillationData> const scintillation;
    NativeRef<PreGenStateData> const state;
    PreGenBufferSize size;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
CELER_FUNCTION void ScintPreGenExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);

    using DistId = ItemId<GeneratorDistributionData>;

    auto tsid = track.track_slot_id();
    CELER_ASSERT(size.scintillation + tsid.get() < state.scintillation.size());
    auto& scintillation_dist
        = state.scintillation[DistId(size.scintillation + tsid.get())];

    // Clear distribution data
    scintillation_dist = {};

    auto sim = track.make_sim_view();
    auto const& step = state.step[tsid];

    if (!step || sim.status() == TrackStatus::inactive)
    {
        // Inactive tracks, materials with no optical properties, or particles
        // that started the step with zero energy (e.g. a stopped positron)
        return;
    }

    Real3 const& pos = track.make_geo_view().pos();
    auto edep = track.make_physics_step_view().energy_deposition();
    auto particle = track.make_particle_view();
    auto rng = track.make_rng_engine();

    // Get the distribution data used to generate scintillation optical photons
    ScintillationPreGenerator generate(
        particle, sim, pos, edep, scintillation, step);
    scintillation_dist = generate(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
