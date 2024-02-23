//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
    auto particle = track.make_particle_view();
    auto const& dir = track.make_geo_view().dir();
    auto rng = track.make_rng_engine();

    // Select a target element
    auto material = track.make_material_view().make_material_view();
    auto elcomp_id = track.make_physics_step_view().element();
    if (!elcomp_id)
    {
        // Sample an element (based on element cross sections on the fly)
        ElementSelector select_el(
            material,
            NeutronElasticMicroXsCalculator{params, particle.energy()},
            track.make_material_view().element_scratch());
        elcomp_id = select_el(rng);
        CELER_ASSERT(elcomp_id);
        track.make_physics_step_view().element(elcomp_id);
    }
    ElementView element = material.make_element_view(elcomp_id);

    // Select a target nucleus
    IsotopeSelector iso_select(element);
    IsotopeView target = element.make_isotope_view(iso_select(rng));

    // Construct the interactor
    ChipsNeutronElasticInteractor interact(params, particle, dir, target);

    // Execute the interactor
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
