//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/launcher/MuBremsstrahlungLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/MuBremsstrahlungData.hh"
#include "celeritas/em/interactor/MuBremsstrahlungInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply MuBremsstrahlung to the current track.
 */
inline CELER_FUNCTION Interaction mu_bremsstrahlung_interact_track(
    MuBremsstrahlungData const& model, CoreTrackView const& track)
{
    auto material_track = track.make_material_view();
    auto material       = material_track.make_material_view();
    auto particle       = track.make_particle_view();
    auto physics        = track.make_physics_view();
    auto rng            = track.make_rng_engine();

    // Sample an element
    auto select_element = physics.make_element_selector(
        physics.action_to_model(model.ids.action), particle.energy());
    auto elcomp_id = select_element(rng);

    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    const auto& dir = track.make_geo_view().dir();

    MuBremsstrahlungInteractor interact(
        model, particle, dir, allocate_secondaries, material, elcomp_id);

    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
