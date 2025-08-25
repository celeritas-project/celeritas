//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfacePhysicsMapBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>

#include "corecel/Types.hh"

#include "SurfaceModel.hh"
#include "SurfacePhysicsMapData.hh"

namespace celeritas
{
class SurfaceParams;
//---------------------------------------------------------------------------//
/*!
 * Create host data for a surface physics map.
 */
class SurfacePhysicsMapBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using HostData = HostVal<SurfacePhysicsMapData>;
    using SurfaceModelId = SurfaceModel::SurfaceModelId;
    //!@}

  public:
    // Construct with surface data and result to modify
    SurfacePhysicsMapBuilder(SurfaceParams const& surfaces, HostData& data);

    // Add and index from a surface model
    void operator()(SurfaceModel const& model);

  private:
    //// TYPES ////

    using InternalSurfaceId = SurfaceModel::InternalSurfaceId;

    //// DATA ////

    //! Geometry-based surface data
    SurfaceParams const& surfaces_;

    //! Data being modified
    HostData& data_;

    //! "Physics surface" for default when user doesn't specify
    SurfaceId default_surface_;

    //! Guard against duplicate IDs
    std::set<SurfaceModelId> surface_models_;

    //// METHODS ////

    //! Reserve an extra "surface" for the default physis
    SurfaceId::size_type size() const { return default_surface_.get() + 1; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
