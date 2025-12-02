//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/GammaNuclearExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/em/data/GammaNuclearData.hh"
#include "celeritas/em/interactor/GammaNuclearInteractor.hh"
#include "celeritas/em/xs/GammaNuclearMicroXsCalculator.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/random/ElementSelector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct GammaNuclearExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    GammaNuclearRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the GammaNuclearInteractor to the current track.
 */
CELER_FUNCTION Interaction
GammaNuclearExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();

    // Select a target element
    auto material = track.material().material_record();
    auto elcomp_id = track.physics_step().element();
    if (!elcomp_id)
    {
        // Sample an element (based on element cross sections on the fly)
        ElementSelector select_el(
            material,
            GammaNuclearMicroXsCalculator{params, particle.energy()},
            track.material().element_scratch());
        elcomp_id = select_el(rng);
        CELER_ASSERT(elcomp_id);
        track.physics_step().element(elcomp_id);
    }

    // Construct the interactor
    GammaNuclearInteractor interact(params, particle);

    // Execute the interactor
    return interact();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
