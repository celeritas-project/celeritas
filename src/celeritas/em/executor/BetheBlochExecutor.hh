//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/BetheBlochExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/BetheBlochData.hh"
#include "celeritas/em/interactor/BetheBlochInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct BetheBlochExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<BetheBlochData> params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the BetheBlochInteractor to the current track.
 */
CELER_FUNCTION Interaction
BetheBlochExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.make_particle_view();
    auto cutoff = track.make_cutoff_view();
    auto const& dir = track.make_geo_view().dir();
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();

    BetheBlochInteractor interact(
        params, particle, cutoff, dir, allocate_secondaries);
    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
