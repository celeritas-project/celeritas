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
 * Populate \c inp::SurfacePhysics data from Geant4 by looping over
 * \c SurfaceIds .
 */
class GeantSurfacePhysicsLoader
{
  public:
    //! Construct with input to be filled by operator()
    GeantSurfacePhysicsLoader(inp::SurfacePhysics& result);

    //! Populate surface physics input data
    void operator()(SurfaceId sid);

  private:
    //// DATA ////
    inp::SurfacePhysics& result_;

    //// HELPER FUNCTIONS ////

    // Insert a given surface to inp::SurfacePhysics::ReflectivityModels
    void insert_reflectivity(detail::GeantSurfacePhysicsHelper& helper);

    // Insert a given surface to inp::SurfacePhysics::RoughnessModels
    void insert_roughness(detail::GeantSurfacePhysicsHelper& helper);

    // Insert a given surface to inp::SurfacePhysics::InteractionModels
    void insert_interaction(detail::GeantSurfacePhysicsHelper& helper);

    // Return true if the surface has *ONLY* analytic reflection
    bool analytic_reflection_only(G4OpticalSurface const& surf) const;

    // Validate model insertion
    void validate_model(detail::GeantSurfacePhysicsHelper& helper) const;
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
