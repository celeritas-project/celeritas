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
    GeantSurfacePhysicsLoader(inp::SurfacePhysics& result);

    //! Populate surface physics data
    void operator()(SurfaceId sid);

  private:
    //// DATA ////
    inp::SurfacePhysics& result_;  // Populated by operator()

    //// HELPER FUNCTIONS ////

    // Insert GLISUR model surface
    void insert_glisur(GeantSurfacePhysicsHelper& helper);

    // Insert Unified model surface
    void insert_unified(GeantSurfacePhysicsHelper& helper);

    // Insert both grid and analytic reflectivities into result_
    void insert_grid_analytic_reflectivities(GeantSurfacePhysicsHelper& helper);
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4
GeantSurfacePhysicsLoader::GeantSurfacePhysicsLoader(inp::SurfacePhysics&) {}

inline void GeantSurfacePhysicsLoader::operator()(SurfaceId) {}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
