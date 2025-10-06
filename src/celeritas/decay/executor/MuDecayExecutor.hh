//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/decay/executor/MuDecayExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/decay/interactor/MuDecayInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct MuDecayExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
/*!
 * Apply the \c MuDecayInteractor to the current track.
 */
CELER_FUNCTION Interaction MuDecayExecutor::operator()(CoreTrackView const& track)
{
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto particle = track.particle();
    auto const& dir = track.geometry().dir();
    auto daughters
        = track.physics().daughters(track.physics_step().decay_channel());

    MuDecayInteractor interact(daughters, particle, dir, allocate_secondaries);
    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
