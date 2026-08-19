//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/executor/ElectroNuclearExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/em/data/ElectroNuclearData.hh"
#include "celeritas/em/interactor/ElectroNuclearInteractor.hh"
#include "celeritas/em/xs/ElectroNuclearMicroXsCalculator.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/mat/ElementView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/random/ElementSelector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct ElectroNuclearExecutor
{
    inline CELER_FUNCTION Interaction operator()(
        celeritas::CoreTrackView const& track);

    NativeCRef<ElectroNuclearData> params;
};

//---------------------------------------------------------------------------//
/*!
 * Apply the ElectroNuclearInteractor to the current track.
 */
CELER_FUNCTION Interaction ElectroNuclearExecutor::operator()(
    CoreTrackView const& track)
{
    auto particle = track.particle();

    // Construct the interactor
    ElectroNuclearInteractor interact(params, particle);

    // Execute the interactor
    return interact();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
