//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/LivermorePEExecutor.hh
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
    auto particle = track.make_particle_view();
    auto rng = track.make_rng_engine();

    // Get the element ID if an element was previously sampled
    auto elcomp_id = track.make_physics_step_view().element();
    if (!elcomp_id)
    {
        // Sample an element (calculating microscopic cross sections on the
        // fly) and store it
        auto material_track = track.make_material_view();
        auto material = material_track.make_material_view();
        ElementSelector select_el(
            material,
            LivermorePEMicroXsCalculator{model, particle.energy()},
            material_track.element_scratch());
        elcomp_id = select_el(rng);
        CELER_ASSERT(elcomp_id);
        track.make_physics_step_view().element(elcomp_id);
    }
    auto el_id = track.make_material_view().make_material_view().element_id(
        elcomp_id);

    // Set up photoelectric interactor with the selected element
    auto relaxation
        = track.make_physics_step_view().make_relaxation_helper(el_id);
    auto cutoffs = track.make_cutoff_view();
    auto const& dir = track.make_geo_view().dir();
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    LivermorePEInteractor interact(
        model, relaxation, el_id, particle, cutoffs, dir, allocate_secondaries);

    // Sample the interaction
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
