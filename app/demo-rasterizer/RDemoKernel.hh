//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RDemoKernel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/GeoData.hh"

#include "ImageData.hh"

namespace demo_rasterizer
{
//---------------------------------------------------------------------------//

using celeritas::MemSpace;
using celeritas::Ownership;

using GeoParamsCRefDevice
    = celeritas::GeoParamsData<Ownership::const_reference, MemSpace::device>;
using GeoStateRefDevice
    = celeritas::GeoStateData<Ownership::reference, MemSpace::device>;

void trace(const GeoParamsCRefDevice& geo_params,
           const GeoStateRefDevice&   geo_state,
           const ImageData&           image);

//---------------------------------------------------------------------------//
} // namespace demo_rasterizer
