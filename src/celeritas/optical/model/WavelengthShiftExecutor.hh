//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"
#include "celeritas/optical/interactor/WavelengthShiftInteractor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct WavelengthShiftExecutor
{
    inline CELER_FUNCTION Interaction operator()(CoreTrackView const&);

    NativeCRef<WavelengthShiftData> data;
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical WLS interaction from the current track.
 */
CELER_FUNCTION Interaction
WavelengthShiftExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto sim = track.sim();
    auto mat_id = track.material_record().material_id();
    auto rng = track.rng();

    WavelengthShiftInteractor interact{
        data, particle, sim, track.geometry().pos(), mat_id};

    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
