//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/launcher/LivermorePELauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/em/interactor/LivermorePEInteractor.hh"
#include "celeritas/em/xs/LivermorePEMicroXsCalculator.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/ElementSelector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply the Livermore photoelectric interaction to the current track.
 */
inline CELER_FUNCTION Interaction livermore_pe_interact_track(
    LivermorePERef const& model, CoreTrackView const& track)
{
    // Set up element sampler
    auto            material = track.make_material_view();
    auto            particle = track.make_particle_view();
    ElementSelector select_el(
        material.make_material_view(),
        LivermorePEMicroXsCalculator{model, particle.energy()},
        material.element_scratch());

    // Sample element
    auto      rng = track.make_rng_engine();
    ElementId el_id;
    {
        ElementComponentId comp_id = select_el(rng);
        CELER_ASSERT(comp_id);
        el_id = material.make_material_view().element_id(comp_id);
    }

    // Set up photoelectric inteactor with the selected element
    auto        relaxation           = track.make_relaxation_helper(el_id);
    auto        cutoffs              = track.make_cutoff_view();
    const auto& dir                  = track.make_geo_view().dir();
    auto        allocate_secondaries = track.make_secondary_allocator();
    LivermorePEInteractor interact(
        model, relaxation, el_id, particle, cutoffs, dir, allocate_secondaries);

    // Sample the interaction
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
