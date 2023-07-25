//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/WentzelExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/WentzelData.hh"
#include "celeritas/em/interactor/WentzelInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/random/RngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct WentzelExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    WentzelRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample Wentzel's model of elastic Coulomb scattering from the current track.
 */
CELER_FUNCTION Interaction WentzelExecutor::operator()(CoreTrackView const& track)
{
    // Incident particle quantities
    auto particle = track.make_particle_view();
    auto const& dir = track.make_geo_view().dir();

    // Material and target quantities
    auto material = track.make_material_view().make_material_view();
    auto elcomp_id = track.make_physics_step_view().element();
    auto cutoffs = track.make_cutoff_view();

    // Construct the interactor
    WentzelInteractor interact(
        params, particle, dir, material, elcomp_id, cutoffs);

    auto rng = track.make_rng_engine();

    // Execute the interactor
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
