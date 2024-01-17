//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/RDemoRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/geo/GeoParams.hh"

#include "ImageStore.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Set up and run rasterization of the given device image.
 */
class RDemoRunner
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<GeoParams const>;
    using Args = ImageRunArgs;
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
}  // namespace app
}  // namespace celeritas
