//---------------------------------*- C++
//-*----------------------------------//
// Copyright ...
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/interactor/MieInteractor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct MieExecutor
{
    inline CELER_FUNCTION Interaction operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Mie interaction from the current track.
 */
CELER_FUNCTION Interaction MieExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto const& direction = track.geometry().dir();
    auto rng = track.rng();

    MieInteractor interact{particle, direction, Params const & mie_params};

    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
