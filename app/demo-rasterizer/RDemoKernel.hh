//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
using GeoStateRefDevice   = celeritas::DeviceRef<celeritas::GeoStateData>;

void trace(const GeoParamsCRefDevice& geo_params,
           const GeoStateRefDevice&   geo_state,
           const ImageData&           image);

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
