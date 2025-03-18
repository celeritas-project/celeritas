//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/KleinNishinaExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/em/data/KleinNishinaData.hh"
#include "celeritas/em/interactor/KleinNishinaInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct KleinNishinaExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    KleinNishinaData params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the KleinNishinaInteractor to the current track.
 */
CELER_FUNCTION Interaction
KleinNishinaExecutor::operator()(CoreTrackView const& track)
{
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto particle = track.particle();
    auto const& dir = track.geometry().dir();

    KleinNishinaInteractor interact(
        params, particle, dir, allocate_secondaries);
    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
