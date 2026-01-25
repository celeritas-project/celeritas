//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/MuPairProductionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/random/engine/RngEngine.hh"
#include "celeritas/em/data/MuPairProductionData.hh"
#include "celeritas/em/interactor/MuPairProductionInteractor.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PhysicsStepView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct MuPairProductionExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    NativeCRef<MuPairProductionData> params;
};

//---------------------------------------------------------------------------//
/*!
 * Sample muon pair production from the current track.
 */
CELER_FUNCTION Interaction
MuPairProductionExecutor::operator()(CoreTrackView const& track)
{
    auto cutoff = track.cutoff();
    auto particle = track.particle();
    auto elcomp_id = track.physics_step().element();
    CELER_ASSERT(elcomp_id);
    auto element = track.material().material_record().element_record(elcomp_id);
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();
    auto const& dir = track.geometry().dir();

    MuPairProductionInteractor interact(
        params, particle, cutoff, element, dir, allocate_secondaries);

    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
