//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/detail/VecgeomParams.cu
//---------------------------------------------------------------------------//
#include "VecGeom/surfaces/cuda/BrepCudaManager.h"

namespace celeritas
{
//---------------------------------------------------------------------------//
using BrepCudaManager = vgbrep::BrepCudaManager<vecgeom::Precision>;
using SurfData = vgbrep::SurfData<vecgeom::Precision>;

void send_surface_data_to_GPU(SurfData const& surfData)
{
    BrepCudaManager::Instance().TransferSurfData(surfData);
}

void cleanup()
{
    BrepCudaManager::Instance().Cleanup();
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
