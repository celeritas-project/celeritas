//---------------------------------*- C++
//-*----------------------------------//
// Copyright ...
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
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
    // NativeCRef<ImportedMaterials> imported; // <-- hold material tables
    // ImportedMaterials const* imported;
    std::vector<ImportMie> const& mie_data;
    inline CELER_FUNCTION Interaction operator()(CoreTrackView const&);
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Mie interaction from the current track.
 */
CELER_FUNCTION Interaction MieExecutor::operator()(CoreTrackView const& track)
{
    CELER_LOG(debug) << "Mie model - Executor called ";
    // Access the current particle track (optical photon)
    auto particle = track.particle();

    // Photon’s current direction
    auto const& direction = track.geometry().dir();

    // RNG stream for sampling scattering
    auto rng = track.rng();

    // Look up the Mie parameters for this material
    auto matid = track.material_record().material_id();
    ImportMie const& mie = mie_data[matid.unchecked_get()];
    CELER_LOG_LOCAL(debug) << "MieExecutor: material=" << matid.get()
                           << " g_forward=" << mie.forward_g
                           << " g_backward=" << mie.backward_g
                           << " forward_ratio=" << mie.forward_ratio;
    //
    //// Construct an interactor that knows how to do Henyey–Greenstein
    /// scattering
    MieInteractor interact{particle, direction, mie};
    return interact(rng);
    //
    //// Run the interaction, producing a scattering event
    // return interact(rng);
    //  Interaction result;
    // return result;
    /*  auto particle = track.particle();
      auto const& direction = track.geometry().dir();
      auto rng = track.rng();

      // Get current material ID
      auto mat_id = track.material_record().material_id();
      CELER_LOG(debug)<<" Mieexec mie interactor";
      // Fetch ImportMie for this material
      //  auto const& import_mie = imported.mie(mat_id);
      // Get imported mie properties for this material
      //auto const& mie_imp = imported->mie(mat_id);

      // Fill MieInteractor::Params
      //MieInteractor::Params params{static_cast<real_type>(mie_imp.forward_g),
      // static_cast<real_type>(mie_imp.backward_g),
      // static_cast<real_type>(mie_imp.forward_ratio)};
  //
      //// Construct and run the interactor
      //MieInteractor interact{particle, direction, params};
    //  return interact(rng);
    return  0;*/
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
