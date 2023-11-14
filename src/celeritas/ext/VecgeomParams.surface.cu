//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/VecgeomParams.surface.cu
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include "corecel/Assert.hh"

#include "VecGeom/surfaces/cuda/BrepCudaManager.h"

namespace celeritas
{
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData = vgbrep::SurfData<vecgeom::Precision>;

//---------------------------------------------------------------------------//
void setup_surface_tracking_device(SurfData const& surfData)
{
    BrepCudaManager::Instance().TransferSurfData(surfData);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
void VecgeomParams::teardown_surface_tracking_device()
{
    BrepCudaManager::Instance().Cleanup();
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
