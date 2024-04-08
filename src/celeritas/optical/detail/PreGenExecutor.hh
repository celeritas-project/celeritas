//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/CerenkovPreGenerator.hh"
#include "celeritas/optical/OpticalGenData.hh"
#include "celeritas/optical/ScintillationPreGenerator.hh"

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
struct PreGenExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<OpticalPropertyData> const properties;
    NativeCRef<CerenkovData> const cerenkov;
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
CELER_FUNCTION void PreGenExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);

    using DistId = ItemId<OpticalDistributionData>;

    auto tsid = track.track_slot_id();
    auto& cerenkov_dist = state.cerenkov[DistId(size.cerenkov + tsid.get())];
    auto& scintillation_dist
        = state.scintillation[DistId(size.scintillation + tsid.get())];

    // clear distribution data
    cerenkov_dist = {};
    scintillation_dist = {};

    auto sim = track.make_sim_view();
    auto optmat_id
        = track.make_material_view().make_material_view().optical_material_id();
    if (sim.status() == TrackStatus::inactive || !optmat_id
        || sim.step_length() == 0)
    {
        // Inactive tracks, materials with no optical properties, or particles
        // that started the step with zero energy (e.g. a stopped positron)
        return;
    }

    Real3 const& pos = track.make_geo_view().pos();
    auto particle = track.make_particle_view();
    auto rng = track.make_rng_engine();

    // Get the distribution data used to generate scintillation and Cerenkov
    // optical photons
    if (cerenkov && particle.charge() != zero_quantity())
    {
        CELER_ASSERT(properties);
        CerenkovPreGenerator generate(particle,
                                      sim,
                                      pos,
                                      optmat_id,
                                      properties,
                                      cerenkov,
                                      state.step[tsid]);
        cerenkov_dist = generate(rng);
    }
    if (scintillation)
    {
        auto edep = track.make_physics_step_view().energy_deposition();
        ScintillationPreGenerator generate(
            particle, sim, pos, optmat_id, edep, scintillation, state.step[tsid]);
        scintillation_dist = generate(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
