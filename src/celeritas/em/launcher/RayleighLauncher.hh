//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/launcher/RayleighLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/RayleighData.hh"
#include "celeritas/em/interactor/RayleighInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply Rayleigh to the current track.
 */
inline CELER_FUNCTION Interaction
rayleigh_interact_track(RayleighRef const& model, CoreTrackView const& track)
{
    const auto& dir      = track.make_geo_view().dir();
    auto        material = track.make_material_view().make_material_view();
    auto        particle = track.make_particle_view();
    auto        physics  = track.make_physics_view();
    auto        rng      = track.make_rng_engine();

    // Sample an element
    auto select_element = physics.make_element_selector(
        physics.action_to_model(model.ids.action), particle.energy());
    auto elcomp_id = select_element(rng);
    auto el_id     = material.element_id(elcomp_id);

    RayleighInteractor interact(model, particle, dir, el_id);
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
