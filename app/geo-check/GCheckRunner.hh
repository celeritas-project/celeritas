//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GCheckRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "geometry/GeoParams.hh"
#include "geometry/GeoInterface.hh"

using namespace celeritas;

namespace geo_check
{
using SPConstGeo = std::shared_ptr<const celeritas::GeoParams>;

//---------------------------------------------------------------------------//
/*!
 * Set up and run rasterization of the given device image.
 */
class GCheckRunner
{
  public:
    //!@{
    //! Type aliases
    using SPConstGeo = std::shared_ptr<const celeritas::GeoParams>;
    //!@}

  public:
    // Construct with geometry
    explicit GCheckRunner(SPConstGeo geometry, int max_steps);

    // Run over some tracks
    void operator()(const celeritas::GeoStateInitializer* init, int ntk) const;

  private:
    SPConstGeo geo_params_;
    int        max_steps_;
};

//---------------------------------------------------------------------------//
} // namespace geo_check
