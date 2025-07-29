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
#include "celeritas/inp/SurfacePhysics.hh"

#include "detail/GeantSurfacePhysicsHelper.hh"

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
    //! Construct empty
    GeantSurfacePhysicsLoader();

    //! Populate surface physics input data
    void operator()(SurfaceId sid, inp::SurfacePhysics& result);

  private:
    //// HELPER FUNCTIONS ////

    // Insert a given surface to inp::SurfacePhysics::ReflectivityModels
    void insert_reflectivity(detail::GeantSurfacePhysicsHelper& helper,
                             inp::SurfacePhysics& result);

    // Insert a given surface to inp::SurfacePhysics::RoughnessModels
    void insert_roughness(detail::GeantSurfacePhysicsHelper& helper,
                          inp::SurfacePhysics& result);

    // Insert a given surface to inp::SurfacePhysics::InteractionModels
    void insert_interaction(detail::GeantSurfacePhysicsHelper& helper,
                            inp::SurfacePhysics& result);

    // Return true if the surface has *ONLY* analytic reflection
    bool analytic_reflection_only(G4OpticalSurface const& surf) const;

    // Validate model insertion
    void validate_model(detail::GeantSurfacePhysicsHelper& helper,
                        inp::SurfacePhysics& result) const;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
inline GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader()
{
    CELER_NOT_CONFIGURED("Geant4");
}

inline inp::SurfacePhysics operator()(SurfaceId, inp::SurfacePhysics&)
{
    CELER_NOT_CONFIGURED("Geant4");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
