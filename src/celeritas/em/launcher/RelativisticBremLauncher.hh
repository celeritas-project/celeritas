//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/launcher/RelativisticBremLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/RelativisticBremData.hh"
#include "celeritas/em/interactor/RelativisticBremInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply RelativisticBrem to the current track.
 */
inline CELER_FUNCTION Interaction relativistic_brem_interact_track(
    RelativisticBremRef const& model, CoreTrackView const& track)
{
    // Select material track view
    auto material = track.make_material_view().make_material_view();

    // Get the sampled element
    auto elcomp_id = track.make_physics_view().element_id();

    auto        particle = track.make_particle_view();
    const auto& dir      = track.make_geo_view().dir();
    auto        allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    auto cutoff = track.make_cutoff_view();

    RelativisticBremInteractor interact(
        model, particle, dir, cutoff, allocate_secondaries, material, elcomp_id);

    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
