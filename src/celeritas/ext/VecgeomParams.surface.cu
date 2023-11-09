//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/VecgeomParams.cu
//---------------------------------------------------------------------------//
//#include "VecgeomParams.hh"
#include "corecel/Assert.hh"

//#include "VecGeom/surfaces/BrepHelper.h"
#include "VecGeom/surfaces/cuda/BrepCudaManager.h"

namespace celeritas
{
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData = vgbrep::SurfData<vecgeom::Precision>;

//---------------------------------------------------------------------------//
/*
template <typename T>
void VecgeomParams::build_surface_tracking_device(T const& data)
{
    //auto const& brep_helper = vgbrep::BrepHelper<real_type>::Instance();
    //auto& cusurf_manager = BrepCudaManager::Instance();
    //cusurf_manager.TransferSurfData(brep_helper.GetSurfData());
    BrepCudaManager::Instance().TransferSurfData(data);
    CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
}
*/

//---------------------------------------------------------------------------//
void build_surface_tracking_device(SurfData const& surfData)
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
