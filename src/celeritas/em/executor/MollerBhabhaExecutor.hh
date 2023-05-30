//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/MollerBhabhaExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/MollerBhabhaData.hh"
#include "celeritas/em/interactor/MollerBhabhaInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Apply MollerBhabha to the current track.
 */
inline CELER_FUNCTION Interaction moller_bhabha_interact_track(
    MollerBhabhaData const& model, CoreTrackView const& track)
{
    auto particle = track.make_particle_view();
    auto cutoff = track.make_cutoff_view();
    auto const& dir = track.make_geo_view().dir();
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();

    MollerBhabhaInteractor interact(
        model, particle, cutoff, dir, allocate_secondaries);
    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
