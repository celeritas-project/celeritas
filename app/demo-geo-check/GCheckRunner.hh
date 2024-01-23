//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geo-check/GCheckRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Assert.hh"
#include "orange/Types.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParams.hh"

namespace celeritas
{
namespace app
{
// using SPConstGeo = std::shared_ptr<const GeoParams>;

//---------------------------------------------------------------------------//
/*!
 * Set up and run rasterization of the given device image.
 */
class GCheckRunner
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    //!@}

  public:
    // Construct with geometry
    explicit GCheckRunner(SPConstGeo const& geometry, int max_steps);

    // Run over some tracks
    void operator()(GeoTrackInitializer const* init) const;

  private:
    SPConstGeo geo_params_;
    int max_steps_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
