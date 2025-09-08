//---------------------------------*- C++
//-*----------------------------------//
// Copyright ...
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/ImportedMaterials.hh"
#include "celeritas/optical/Interaction.hh"
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

    // NativeCRef<ImportedMaterials> imported; // <-- hold material tables
    ImportedMaterials const* imported;
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Mie interaction from the current track.
 */
CELER_FUNCTION Interaction MieExecutor::operator()(CoreTrackView const& track)
{
    auto particle = track.particle();
    auto const& direction = track.geometry().dir();
    auto rng = track.rng();

    // Get current material ID
    auto mat_id = track.material_record().material_id();

    // Fetch ImportMie for this material
    //  auto const& import_mie = imported.mie(mat_id);
    // Get imported mie properties for this material
    auto const& mie_imp = imported->mie(mat_id);

    // Fill MieInteractor::Params
    MieInteractor::Params params{static_cast<real_type>(mie_imp.forward_g),
                                 static_cast<real_type>(mie_imp.backward_g),
                                 static_cast<real_type>(mie_imp.forward_ratio)};

    // Construct and run the interactor
    MieInteractor interact{particle, direction, params};
    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
