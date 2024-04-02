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
#include "corecel/data/StackAllocator.hh"
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

    using DistributionAllocator = StackAllocator<OpticalDistributionData>;

    auto optmat_id
        = track.make_material_view().make_material_view().optical_material_id();
    if (!optmat_id)
    {
        // No optical properties for this material
        return;
    }

    auto particle = track.make_particle_view();
    auto sim = track.make_sim_view();
    Real3 const& pos = track.make_geo_view().pos();
    auto rng = track.make_rng_engine();

    // Total number of scintillation and Cerenkov photons to generate
    // TODO: Do we return pointer to the distribution data, number of photons
    // to generate? Best way to do reduction?
    size_type num_photons{0};
    if (cerenkov)
    {
        CELER_ASSERT(properties);
        DistributionAllocator allocate{state.cerenkov};
        CerenkovPreGenerator generate(particle,
                                      sim,
                                      pos,
                                      optmat_id,
                                      properties,
                                      cerenkov,
                                      state.step[track.track_slot_id()],
                                      allocate);
        num_photons += generate(rng);
    }
    if (scintillation)
    {
        auto edep = track.make_physics_step_view().energy_deposition();

        DistributionAllocator allocate{state.scintillation};
        ScintillationPreGenerator generate(particle,
                                           sim,
                                           pos,
                                           optmat_id,
                                           edep,
                                           scintillation,
                                           state.step[track.track_slot_id()],
                                           allocate);
        num_photons += generate(rng);
    }
    (void)num_photons;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
