//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct MollerBhabhaExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    MollerBhabhaData params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample Moller-Bhabha ionization from the current track.
 */
CELER_FUNCTION Interaction
MollerBhabhaExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto cutoff = track.cutoff();
    auto const& dir = track.geometry().dir();
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();

    MollerBhabhaInteractor interact(
        params, particle, cutoff, dir, allocate_secondaries);
    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
