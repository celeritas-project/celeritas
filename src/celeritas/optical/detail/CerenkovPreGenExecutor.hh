//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/CerenkovPreGenExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/optical/CerenkovPreGenerator.hh"
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
struct CerenkovPreGenExecutor
{
    inline CELER_FUNCTION void
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<OpticalPropertyData> const properties;
    NativeCRef<CerenkovData> const cerenkov;
    NativeRef<OpticalGenStateData> const state;
    OpticalBufferSize size;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
CELER_FUNCTION void
CerenkovPreGenExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(state);
    CELER_EXPECT(cerenkov);
    CELER_EXPECT(properties);

    using DistId = ItemId<OpticalDistributionData>;

    auto tsid = track.track_slot_id();
    CELER_ASSERT(size.cerenkov + tsid.get() < state.cerenkov.size());
    auto& cerenkov_dist = state.cerenkov[DistId(size.cerenkov + tsid.get())];

    // Clear distribution data
    cerenkov_dist = {};

    auto sim = track.make_sim_view();
    auto step = state.step[tsid];

    if (!step || sim.status() == TrackStatus::inactive)
    {
        // Inactive tracks, materials with no optical properties, or particles
        // that started the step with zero energy (e.g. a stopped positron)
        return;
    }

    auto particle = track.make_particle_view();

    // Get the distribution data used to generate Cerenkov optical photons
    if (particle.charge() != zero_quantity())
    {
        Real3 const& pos = track.make_geo_view().pos();
        auto rng = track.make_rng_engine();

        CerenkovPreGenerator generate(
            particle, sim, pos, properties, cerenkov, step);
        cerenkov_dist = generate(rng);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
