//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.cc
//---------------------------------------------------------------------------//
#include "GeoParams.hh"

#include <iostream>

#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <celeritas_config.h>
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#    include <VecGeom/management/CudaManager.h>
#endif

#include "base/Stopwatch.hh"
#include "base/ColorUtils.hh"
#include "GeoParamsPointers.hh"

using std::cerr;
using std::endl;

namespace celeritas
{
//---------------------------------------------------------------------------//
// MANAGEMENT
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
GeoParams::GeoParams(const char* gdml_filename)
{
    cerr << "::: Loading from GDML at " << gdml_filename << "..." << std::flush;
    constexpr bool validate_xml_schema = false;
    Stopwatch      get_time;
    vgdml::Frontend::Load(gdml_filename, validate_xml_schema);
    cerr << color_code('x') << " (" << get_time() << " s)" << color_code(' ')
         << endl;

    cerr << "::: Initializing tracking information" << endl;
    vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();

    num_volumes_ = vecgeom::VPlacedVolume::GetIdCount();
    max_depth_   = vecgeom::GeoManager::Instance().getMaxDepth();

#if CELERITAS_USE_CUDA
    cerr << "::: Loading geometry..." << std::flush;
    get_time           = {};
    auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
    // cuda_manager.set_verbose(1);
    cuda_manager.LoadGeometry();
    CELER_CUDA_CALL(cudaDeviceSynchronize());
    cerr << color_code('x') << " (" << get_time() << " s)" << color_code(' ')
         << endl;

    get_time = {};
    cerr << "::: Transferring geometry to GPU..." << std::flush;
    auto world_top_devptr = cuda_manager.Synchronize();
    cerr << color_code('x') << " (" << get_time() << " s)" << color_code(' ')
         << endl;

    CHECK(world_top_devptr != nullptr);
    device_world_volume_ = world_top_devptr.GetPtr();
    CELER_CUDA_CHECK_ERROR();
    cerr << ">>> Synchronized successfully!" << endl;

    ENSURE(device_world_volume_);
#endif
    ENSURE(num_volumes_ > 0);
    ENSURE(max_depth_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up vecgeom on destruction
 */
GeoParams::~GeoParams()
{
#if CELERITAS_USE_CUDA
    vecgeom::CudaManager::Instance().CleanGpu();
#endif
    vecgeom::GeoManager::Instance().Clear();
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a placed volume ID
 */
const std::string& GeoParams::id_to_label(VolumeId vol_id) const
{
    REQUIRE(vol_id.get() < num_volumes_);
    const auto* vol
        = vecgeom::GeoManager::Instance().FindPlacedVolume(vol_id.get());
    CHECK(vol);
    return vol->GetLabel();
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label
 */
auto GeoParams::label_to_id(const std::string& label) const -> VolumeId
{
    const auto* vol
        = vecgeom::GeoManager::Instance().FindPlacedVolume(label.c_str());
    CHECK(vol);
    CHECK(vol->id() < num_volumes_);
    return VolumeId{vol->id()};
}

//---------------------------------------------------------------------------//
/*!
 * View in-host geometry data for CPU debugging.
 */
GeoParamsPointers GeoParams::host_view() const
{
    GeoParamsPointers result;
    result.world_volume = vecgeom::GeoManager::Instance().GetWorld();
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed on-device data.
 */
GeoParamsPointers GeoParams::device_pointers() const
{
    REQUIRE(device_world_volume_);
    GeoParamsPointers result;
    result.world_volume
        = static_cast<const vecgeom::VPlacedVolume*>(device_world_volume_);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Increase CUDA stack size to enable complex geometries.
 *
 * For example, CMS requires a stack limit of at least 8192 * 4.
 */
void GeoParams::set_cuda_stack_size(int limit)
{
    REQUIRE(limit > 0);
#if CELERITAS_USE_CUDA
    CELER_CUDA_CALL(cudaDeviceSetLimit(cudaLimitStackSize, limit));
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
