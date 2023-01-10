//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-rasterizer/RDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/geo/GeoData.hh"

#include "ImageData.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//

using GeoParamsCRefDevice = celeritas::DeviceCRef<celeritas::GeoParamsData>;
using GeoStateRefDevice = celeritas::DeviceRef<celeritas::GeoStateData>;

void trace(GeoParamsCRefDevice const& geo_params,
           GeoStateRefDevice const& geo_state,
           ImageData const& image);

//---------------------------------------------------------------------------//
}  // namespace demo_rasterizer
