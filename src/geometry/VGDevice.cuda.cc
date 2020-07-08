//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGDevice.cc
//---------------------------------------------------------------------------//
#include "VGDevice.hh"

#include <cuda_runtime_api.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include "VGView.hh"

using std::cout;
using std::endl;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host geometry.
 *
 * The constructor synchronizes the geometry to the device.
 */
VGDevice::VGDevice(constSPVGHost host_geom) : host_geom_(std::move(host_geom))
{
    REQUIRE(host_geom_);
    cout << "::: Transferring geometry to GPU" << endl;
    auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
    cuda_manager.set_verbose(1);
    cuda_manager.LoadGeometry();
    CELER_CUDA_CHECK_ERROR();

    auto world_top_devptr = cuda_manager.Synchronize();
    CHECK(world_top_devptr != nullptr);
    CELER_CUDA_CHECK_ERROR();
    cout << ">>> Synchronized successfully!" << endl;

    cuda_manager.PrintGeometry();
    device_world_volume_ = world_top_devptr.GetPtr();
    ENSURE(device_world_volume_);
}

//---------------------------------------------------------------------------//
/*!
 * Access on-device geometry data.
 */
VGView VGDevice::device_view() const
{
    REQUIRE(device_world_volume_);
    VGView result;
    result.world_volume
        = static_cast<const vecgeom::VPlacedVolume*>(device_world_volume_);
    return result;
}
//---------------------------------------------------------------------------//
} // namespace celeritas
