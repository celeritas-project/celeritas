//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/launcher/CombinedBremLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/CombinedBremData.hh"
#include "celeritas/em/interactor/CombinedBremInteractor.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/random/RngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply electron/positron brems to the current track.
 */
inline CELER_FUNCTION Interaction combined_brem_interact_track(
    CombinedBremRef const& model, CoreTrackView const& track)
{
    // Select material track view
    auto material = track.make_material_view().make_material_view();

    // Assume only a single element in the material, for now
    CELER_ASSERT(material.num_elements() == 1);
    const ElementComponentId selected_element{0};

    auto particle = track.make_particle_view();
    auto const& dir = track.make_geo_view().dir();
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    auto cutoff = track.make_cutoff_view();

    CombinedBremInteractor interact(model,
                                    particle,
                                    dir,
                                    cutoff,
                                    allocate_secondaries,
                                    material,
                                    selected_element);

    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
