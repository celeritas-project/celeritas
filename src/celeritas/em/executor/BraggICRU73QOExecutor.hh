//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/BraggICRU73QOExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/BraggICRU73QOData.hh"
#include "celeritas/em/interactor/BraggICRU73QOInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct BraggICRU73QOExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    BraggICRU73QOData params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the BraggICRU73QOInteractor to the current track.
 */
CELER_FUNCTION Interaction
BraggICRU73QOExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.make_particle_view();
    auto cutoff = track.make_cutoff_view();
    auto const& dir = track.make_geo_view().dir();
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();

    BraggICRU73QOInteractor interact(
        params, particle, cutoff, dir, allocate_secondaries);
    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
