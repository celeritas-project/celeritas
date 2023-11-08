//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/VecgeomParams.cu
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include "VecGeom/surfaces/cuda/BrepCudaManager.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
void VecgeomParams::build_surface_tracking_device()
{
    // auto const& brep_helper = vgbrep::BrepHelper<real_type>::Instance();
    // auto& cusurf_manager = CudaSurfManager::Instance();
    // cusurf_manager.TransferSurfData(brep_helper.GetSurfData());
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

//---------------------------------------------------------------------------//
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData = vgbrep::SurfData<vecgeom::Precision>;

void send_surface_data_to_GPU(SurfData const& surfData)
{
    BrepCudaManager::Instance().TransferSurfData(surfData);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}

void cleanup_surface_data_gpu()
{
    BrepCudaManager::Instance().Cleanup();
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
