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
    //!@}

  public:
    // Construct with default surface ID and result to modify
    SurfacePhysicsMapBuilder(PhysSurfaceId::size_type num_surfaces,
                             HostData& data);

    // Add and index from a surface model
    void operator()(SurfaceModel const& model);

  private:
    //// DATA ////

    //! Data being modified
    HostData& data_;

    //! Guard against duplicate IDs
    std::set<SurfaceModelId> surface_models_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
