//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/VecgeomParams.surface.cu
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include <VecGeom/surfaces/cuda/BrepCudaManager.h>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void setup_surface_tracking_device(VGSurfData const& surf_data)
{
    VGBrepCudaManager::Instance().TransferSurfData(surf_data);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
void teardown_surface_tracking_device()
{
    VGBrepCudaManager::Instance().Cleanup();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
