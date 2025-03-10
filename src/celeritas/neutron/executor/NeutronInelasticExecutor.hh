//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/executor/NeutronInelasticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/ElementSelector.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/neutron/data/NeutronInelasticData.hh"
#include "celeritas/neutron/interactor/NeutronInelasticInteractor.hh"
#include "celeritas/neutron/xs/NeutronInelasticMicroXsCalculator.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct NeutronInelasticExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    NeutronInelasticRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the NeutronInelasticInteractor to the current track.
 */
CELER_FUNCTION Interaction
NeutronInelasticExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto rng = track.rng();

    // Select a target element
    auto material = track.material().material_record();
    auto elcomp_id = track.physics_step().element();
    if (!elcomp_id)
    {
        // Sample an element (based on element cross sections on the fly)
        ElementSelector select_el(
            material,
            NeutronInelasticMicroXsCalculator{params, particle.energy()},
            track.material().element_scratch());
        elcomp_id = select_el(rng);
        CELER_ASSERT(elcomp_id);
        track.physics_step().element(elcomp_id);
    }

    // Construct the interactor
    NeutronInelasticInteractor interact(params, particle);

    // Execute the interactor
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
