//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "celeritas/inp/SurfacePhysics.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

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
    GeantSurfacePhysicsLoader(inp::SurfacePhysics& surface_phys,
                              std::vector<ImportOpticalMaterial>& materials);

    //! Populate surface physics data
    void operator()(SurfaceId sid);

  private:
    //// DATA ////
    inp::SurfacePhysics& models_;  // Populated by operator()
    std::vector<ImportOpticalMaterial>& materials_;  // Populated by operator()

    PhysSurfaceId current_surface_{0};

    //// HELPER FUNCTIONS ////

    // Insert a value for the current surface into a model map in place
    template<class T>
    void emplace(std::map<PhysSurfaceId, T>& m, T&& value);

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

    // Insert gap material and surface for back-painted surfaces
    void insert_gap_material(GeantSurfacePhysicsHelper const& helper);

    // Insert painted surface (reflection only)
    void insert_painted_surface(optical::ReflectionMode mode);
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader(inp::SurfacePhysics&) {}

inline void GeantSurfacePhysicsLoader::operator()(SurfaceId) {}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
