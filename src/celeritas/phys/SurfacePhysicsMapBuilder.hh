//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfacePhysicsMapBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <set>

#include "corecel/Types.hh"

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
    //!@}

  public:
    // Construct with surface data and result to modify
    SurfacePhysicsMapBuilder(SurfaceParams const& surfaces, HostData& data);

    // Add and index from a surface model
    void operator()(SurfaceModel const& model);

  private:
    SurfaceParams const& surfaces_;
    HostData& data_;
    std::set<ActionId> actions_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
