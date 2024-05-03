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
#include "celeritas/optical/OpticalGenData.hh"
#include "celeritas/optical/ScintillationPreGenerator.hh"

#include "Utils.hh"

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
struct ScintPreGenExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<ScintillationData> const scintillation;
    NativeRef<OpticalGenStateData> const state;
    OpticalBufferSize size;
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

    using DistId = ItemId<OpticalDistributionData>;

    auto tsid = track.track_slot_id();
    CELER_ASSERT(size.scintillation + tsid.get() < state.scintillation.size());
    auto& scintillation_dist
        = state.scintillation[DistId(size.scintillation + tsid.get())];

    // Clear distribution data
    scintillation_dist = {};

    auto sim = track.make_sim_view();
    auto optmat_id = get_optical_material(track);
    if (!optmat_id || sim.step_length() == 0)
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
        particle, sim, pos, optmat_id, edep, scintillation, state.step[tsid]);
    scintillation_dist = generate(rng);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
