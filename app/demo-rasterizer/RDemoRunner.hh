//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RDemoRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include "geometry/GeoParams.hh"
#include "ImageStore.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//
/*!
 * Set up and run rasterization of the given device image.
 */
class RDemoRunner
{
  public:
    //!@{
    //! Type aliases
    using SPConstGeo = std::shared_ptr<const celeritas::GeoParams>;
    using Args       = ImageRunArgs;
    //!@}

  public:
    // Construct with geometry
    explicit RDemoRunner(SPConstGeo geometry);

    // Trace an image
    void operator()(ImageStore* image, int ntimes = 0) const;

  private:
    SPConstGeo geo_params_;
};

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
