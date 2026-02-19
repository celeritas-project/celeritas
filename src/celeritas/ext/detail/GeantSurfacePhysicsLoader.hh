//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "celeritas/inp/SurfacePhysics.hh"

#include "GeantSurfacePhysicsHelper.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Populate \c inp::SurfacePhysics data from Geant4 by looping over
 * \c SurfaceIds .
 */
class GeantSurfacePhysicsLoader
{
  public:
    //! Construct with \c SurfacePhysics input
    GeantSurfacePhysicsLoader(inp::SurfacePhysics& surface_phys);

    //! Populate surface physics data
    void operator()(SurfaceId sid);

  private:
    //// DATA ////
    inp::SurfacePhysics& models_;  // Populated by operator()

    //// HELPER FUNCTIONS ////

    // Check that unimplemented properties are not present
    void check_unimplemented_properties(
        GeantSurfacePhysicsHelper const& helper) const;

    // Insert GLISUR model surface
    void insert_glisur(GeantSurfacePhysicsHelper const& helper);

    // Insert Unified model surface
    void insert_unified(GeantSurfacePhysicsHelper const& helper);

    // Insert grid or analytic reflectivity into models_
    void insert_reflectivity(GeantSurfacePhysicsHelper const& helper);

    // Insert reflection form for di/di or di/met
    void insert_interaction(GeantSurfacePhysicsHelper const& helper,
                            inp::ReflectionForm&& rf);
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader(inp::SurfacePhysics&) {}

inline void GeantSurfacePhysicsLoader::operator()(SurfaceId) {}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
