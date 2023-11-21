//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomParams.surface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/surfaces/BrepHelper.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
using VGBrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using VGSurfData = vgbrep::SurfData<vecgeom::Precision>;

// Set up surface tracking
void setup_surface_tracking_device(VGSurfData const& surf_data);

// Tear down surface tracking
void teardown_surface_tracking_device();

//---------------------------------------------------------------------------//
}  // namespace celeritas
