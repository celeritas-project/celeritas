//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/executor/ChipsNeutronElasticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/ElementSelector.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/mat/IsotopeSelector.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/neutron/data/NeutronElasticData.hh"
#include "celeritas/neutron/interactor/ChipsNeutronElasticInteractor.hh"
#include "celeritas/neutron/xs/NeutronElasticMicroXsCalculator.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct ChipsNeutronElasticExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    NeutronElasticRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the NeutronElasticInteractor to the current track.
 */
CELER_FUNCTION Interaction
ChipsNeutronElasticExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto const& dir = track.geometry().dir();
    auto rng = track.rng();

    // Select a target element
    auto material = track.material().material_record();
    auto elcomp_id = track.physics_step().element();
    if (!elcomp_id)
    {
        // Sample an element (based on element cross sections on the fly)
        ElementSelector select_el(
            material,
            NeutronElasticMicroXsCalculator{params, particle.energy()},
            track.material().element_scratch());
        elcomp_id = select_el(rng);
        CELER_ASSERT(elcomp_id);
        track.physics_step().element(elcomp_id);
    }
    ElementView element = material.element_record(elcomp_id);

    // Select a target nucleus
    IsotopeSelector iso_select(element);
    IsotopeView target = element.isotope_record(iso_select(rng));

    // Construct the interactor
    ChipsNeutronElasticInteractor interact(params, particle, dir, target);

    // Execute the interactor
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
