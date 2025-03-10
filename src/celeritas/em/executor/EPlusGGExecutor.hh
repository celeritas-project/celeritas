//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto particle = track.particle();
    auto const& dir = track.geometry().dir();

    EPlusGGInteractor interact(params, particle, dir, allocate_secondaries);
    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
