//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    auto material = track.material().material_record();
    auto particle = track.particle();
    auto elcomp_id = track.physics_step().element();
    CELER_ASSERT(elcomp_id);
    auto element = material.element_record(elcomp_id);
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto const& dir = track.geometry().dir();

    BetheHeitlerInteractor interact(
        params, particle, dir, allocate_secondaries, material, element);

    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
