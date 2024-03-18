//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/CoulombScatteringExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/CoulombScatteringData.hh"
#include "celeritas/em/interactor/CoulombScatteringInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/IsotopeSelector.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/random/RngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct CoulombScatteringExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    CoulombScatteringRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample Wentzel's model of elastic Coulomb scattering from the current track.
 */
CELER_FUNCTION Interaction
CoulombScatteringExecutor::operator()(CoreTrackView const& track)
{
    // Incident particle quantities
    auto particle = track.make_particle_view();
    auto const& dir = track.make_geo_view().dir();

    // Material and target quantities
    auto material = track.make_material_view().make_material_view();
    auto elcomp_id = track.make_physics_step_view().element();
    auto element_id = material.element_id(elcomp_id);
    auto cutoffs = track.make_cutoff_view();

    auto rng = track.make_rng_engine();

    // Select isotope
    ElementView element = material.make_element_view(elcomp_id);
    IsotopeSelector iso_select(element);
    IsotopeView target = element.make_isotope_view(iso_select(rng));

    // Construct the interactor
    CoulombScatteringInteractor interact(
        params, particle, dir, target, element_id, cutoffs);

    // Execute the interactor
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
