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
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply MuBremsstrahlung to the current track.
 */
inline CELER_FUNCTION Interaction mu_bremsstrahlung_interact_track(
    MuBremsstrahlungData const& model, CoreTrackView const& track)
{
    // Select material track view
    auto material_track = track.make_material_view();
    auto material       = material_track.make_material_view();

    // Assume only a single element in the material, for now
    CELER_ASSERT(material.num_elements() == 1);
    const ElementComponentId elcomp_id{0};

    auto        allocate_secondaries = track.make_secondary_allocator();
    auto        particle             = track.make_particle_view();
    const auto& dir                  = track.make_geo_view().dir();

    MuBremsstrahlungInteractor interact(
        model, particle, dir, allocate_secondaries, material, elcomp_id);

    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
