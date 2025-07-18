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
                             G4OpticalSurface const& surf,
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

    // Insert a given surface to inp::SurfacePhysics::DetectionEfficiency
    void insert_efficiency(SurfaceId sid,
                           detail::GeantMaterialPropertyGetter& get_property,
                           inp::SurfacePhysics& result);

    // Return true if the surface has *ONLY* analytic reflection
    bool analytic_reflection_only(G4OpticalSurface const& surf) const;

    // Calculate the diffuse lobe from the other ReflectionForm properties
    inp::Grid calc_diffuse_lobe(inp::ReflectionForm const& refl_form);
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
