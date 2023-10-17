//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/BetheHeitlerExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/BetheHeitlerData.hh"
#include "celeritas/em/interactor/BetheHeitlerInteractor.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/random/RngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct BetheHeitlerExecutor
{
    inline static constexpr int max_block_size{224};
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    BetheHeitlerData params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample a Bethe-Heitler pair production from the current track.
 */
CELER_FUNCTION Interaction
BetheHeitlerExecutor::operator()(CoreTrackView const& track)
{
    auto material_track = track.make_material_view();
    auto material = material_track.make_material_view();
    auto particle = track.make_particle_view();

    auto elcomp_id = track.make_physics_step_view().element();
    CELER_ASSERT(elcomp_id);
    auto element = material.make_element_view(elcomp_id);
    auto allocate_secondaries
        = track.make_physics_step_view().make_secondary_allocator();
    auto const& dir = track.make_geo_view().dir();

    BetheHeitlerInteractor interact(
        params, particle, dir, allocate_secondaries, material, element);

    auto rng = track.make_rng_engine();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
