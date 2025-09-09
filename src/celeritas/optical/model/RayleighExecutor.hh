//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/interactor/RayleighInteractor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct RayleighExecutor
{
    inline CELER_FUNCTION Interaction operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Rayleigh interaction from the current track.
 */
CELER_FUNCTION Interaction
RayleighExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto const& direction = track.geometry().dir();
    auto rng = track.rng();

    RayleighInteractor interact{particle, direction};

    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
