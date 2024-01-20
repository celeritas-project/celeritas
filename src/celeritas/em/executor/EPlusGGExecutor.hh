//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/EPlusGGExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/interactor/EPlusGGInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct EPlusGGExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    EPlusGGData params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample a positron annihilation from the current track.
 */
CELER_FUNCTION Interaction EPlusGGExecutor::operator()(CoreTrackView const& track)
{
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    auto particle = track.make_particle_view();
    auto const& dir = track.make_geo_view().dir();

    EPlusGGInteractor interact(params, particle, dir, allocate_secondaries);
    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
