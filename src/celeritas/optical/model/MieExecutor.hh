//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/ImportedMaterials.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/MieData.hh"
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

    NativeCRef<MieData> data;
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Mie interaction from the current track.
 */
CELER_FUNCTION Interaction MieExecutor::operator()(CoreTrackView const& track)
{
    // Access the current particle track (optical photon)
    auto particle = track.particle();
    // Photon’s current direction
    auto const& direction = track.geometry().dir();
    // RNG stream for sampling scattering
    auto rng = track.rng();
    // Look up the Mie parameters for this material
    auto matid = track.material_record().material_id();

    MieInteractor interact{data, particle, direction, matid};

    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
