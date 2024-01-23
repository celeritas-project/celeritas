//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/RDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/geo/GeoData.hh"

#include "ImageData.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//

using GeoParamsCRefDevice = DeviceCRef<GeoParamsData>;
using GeoStateRefDevice = DeviceRef<GeoStateData>;

void trace(GeoParamsCRefDevice const& geo_params,
           GeoStateRefDevice const& geo_state,
           ImageData const& image);

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
