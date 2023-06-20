//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/WokviExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/WokviData.hh"
#include "celeritas/em/interactor/WokviInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/random/RngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct WokviExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    WokviRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample Seltzer-Berger bremsstrahlung from the current track.
 */
CELER_FUNCTION Interaction WokviExecutor::operator()(CoreTrackView const& track)
{
    auto material = track.make_material_view().make_material_view();
    auto particle = track.make_particle_view();

    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    auto elcomp_id = track.make_physics_step_view().element();

    auto const& dir = track.make_geo_view().dir();

    WokviInteractor interact(
        params, particle, dir, material, elcomp_id, allocate_secondaries);

    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
