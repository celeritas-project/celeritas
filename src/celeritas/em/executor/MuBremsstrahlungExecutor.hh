//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/MuBremsstrahlungExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "celeritas/em/data/MuBremsstrahlungData.hh"
#include "celeritas/em/interactor/MuBremsstrahlungInteractor.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct MuBremsstrahlungExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    MuBremsstrahlungData params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample MuBremsstrahlung from the current track.
 */
CELER_FUNCTION Interaction
MuBremsstrahlungExecutor::operator()(CoreTrackView const& track)
{
    auto cutoff = track.cutoff();
    auto material = track.material().material_record();
    auto particle = track.particle();

    auto elcomp_id = track.physics_step().element();
    CELER_ASSERT(elcomp_id);
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto const& dir = track.geometry().dir();

    MuBremsstrahlungInteractor interact(
        params, particle, dir, cutoff, allocate_secondaries, material, elcomp_id);

    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
