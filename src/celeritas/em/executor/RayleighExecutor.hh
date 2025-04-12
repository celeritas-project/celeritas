//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/RayleighExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "celeritas/em/data/RayleighData.hh"
#include "celeritas/em/interactor/RayleighInteractor.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RayleighExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    RayleighRef params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample Rayleigh scattering from the current track.
 */
CELER_FUNCTION Interaction
RayleighExecutor::operator()(CoreTrackView const& track)
{
    auto material = track.material().material_record();
    auto particle = track.particle();

    auto elcomp_id = track.physics_step().element();
    CELER_ASSERT(elcomp_id);
    auto el_id = material.element_id(elcomp_id);
    auto const& dir = track.geometry().dir();

    RayleighInteractor interact(params, particle, dir, el_id);

    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
