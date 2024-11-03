//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VecgeomSetup.cu
//---------------------------------------------------------------------------//
#include "VecgeomSetup.cuda.hh"

#ifdef VECGEOM_USE_SURF
#    include <VecGeom/surfaces/cuda/BrepCudaManager.h>
#endif

#include "corecel/Assert.hh"

#ifdef VECGEOM_USE_SURF
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData = vgbrep::SurfData<vecgeom::Precision>;
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// VECGEOM SURFACE
//---------------------------------------------------------------------------//
#ifdef VECGEOM_USE_SURF
void setup_surface_tracking_device(SurfData const& surf_data)
{
    BrepCudaManager::Instance().TransferSurfData(surf_data);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

void teardown_surface_tracking_device()
{
    BrepCudaManager::Instance().Cleanup();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
