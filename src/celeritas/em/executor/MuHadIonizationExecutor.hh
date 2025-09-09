//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/MuHadIonizationExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/em/data/MuHadIonizationData.hh"
#include "celeritas/em/interactor/MuHadIonizationInteractor.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class EnergySampler>
struct MuHadIonizationExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(celeritas::CoreTrackView const& track);

    MuHadIonizationData params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the muon or hadron ionization interactor to the current track.
 */
template<class ES>
CELER_FUNCTION Interaction
MuHadIonizationExecutor<ES>::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto cutoff = track.cutoff();
    auto const& dir = track.geometry().dir();
    auto allocate_secondaries = track.physics_step().make_secondary_allocator();

    MuHadIonizationInteractor<ES> interact(
        params, particle, cutoff, dir, allocate_secondaries);
    auto rng = track.rng();
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
