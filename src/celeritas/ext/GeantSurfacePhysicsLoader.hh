//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSurfacePhysicsLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "geocel/GeantGeoParams.hh"
#include "celeritas/ext/detail/GeantMaterialPropertyGetter.hh"
#include "celeritas/inp/Surfaces.hh"

// Geant4 forward declaration
class G4OpticalSurface;  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 */
class GeantSurfacePhysicsLoader
{
  public:
    //! Construct expecting \c GeantGeoParams available in memory
    GeantSurfacePhysicsLoader();

    // Create surface physics input data
    inp::SurfacePhysics operator()();

  private:
    //// DATA ////
    GeantGeoParams const* geo_;

    //// HELPER FUNCTIONS ////

    // Insert a given surface to inp::SurfacePhysics::ReflectivityModels
    void insert_reflectivity(SurfaceId sid,
                             detail::GeantMaterialPropertyGetter& get_property,
                             inp::SurfacePhysics& result);

    // Insert a given surface to inp::SurfacePhysics::RoughnessModels
    void insert_roughness(SurfaceId sid,
                          G4OpticalSurface& surf,
                          inp::SurfacePhysics& result);

    // Insert a given surface to inp::SurfacePhysics::InteractionModels
    void insert_interaction(SurfaceId sid,
                            detail::GeantMaterialPropertyGetter& get_property,
                            G4OpticalSurface& surf,
                            inp::SurfacePhysics& result);
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader()
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline inp::SurfacePhysics operator()()
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
