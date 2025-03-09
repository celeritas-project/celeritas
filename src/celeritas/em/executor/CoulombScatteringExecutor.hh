//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/CoulombScatteringExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/CoulombScatteringData.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"
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

    CoulombScatteringData params;
    NativeCRef<WentzelOKVIData> wentzel;
};

//---------------------------------------------------------------------------//
/*!
 * Sample Wentzel's model of elastic Coulomb scattering from the current track.
 */
CELER_FUNCTION Interaction
CoulombScatteringExecutor::operator()(CoreTrackView const& track)
{
    // Incident particle quantities
    auto particle = track.particle();
    auto const& dir = track.geometry().dir();

    // Material and target quantities
    auto material = track.material().material_record();
    auto elcomp_id = track.physics_step().element();
    auto element_id = material.element_id(elcomp_id);
    auto cutoffs = track.cutoff();

    auto rng = track.rng();

    // Select isotope
    ElementView element = material.element_record(elcomp_id);
    IsotopeSelector iso_select(element);
    IsotopeView target = element.isotope_record(iso_select(rng));

    // Construct the interactor
    CoulombScatteringInteractor interact(
        params, wentzel, particle, dir, material, target, element_id, cutoffs);

    // Execute the interactor
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
