//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.cc
//---------------------------------------------------------------------------//
#include "GeoParams.hh"

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
#include "comm/Logger.hh"
#include "GeoParamsPointers.hh"

namespace
{
//---------------------------------------------------------------------------//
void print_time(double time_sec)
{
    using celeritas::color_code;

    if (time_sec > 0.01)
    {
        CELER_LOG(diagnostic) << color_code('x') << "... " << time_sec << " s"
                              << color_code(' ');
    }
}
//---------------------------------------------------------------------------//
} // namespace

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
    CELER_LOG(info) << "Loading from GDML at " << gdml_filename;
    constexpr bool validate_xml_schema = false;
    Stopwatch      get_time;
    vgdml::Frontend::Load(gdml_filename, validate_xml_schema);
    print_time(get_time());

    CELER_LOG(status) << "Initializing tracking information";
    get_time = {};
    vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    print_time(get_time());

    num_volumes_ = vecgeom::VPlacedVolume::GetIdCount();
    max_depth_   = vecgeom::GeoManager::Instance().getMaxDepth();

#if CELERITAS_USE_CUDA
    CELER_LOG(status) << "Converting to CUDA geometry";
    get_time           = {};
    auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
    // cuda_manager.set_verbose(1);
    cuda_manager.LoadGeometry();
    CELER_CUDA_CALL(cudaDeviceSynchronize());
    print_time(get_time());

    CELER_LOG(status) << "Transferring geometry to GPU";
    get_time = {};
    auto world_top_devptr = cuda_manager.Synchronize();
    CHECK(world_top_devptr != nullptr);
    device_world_volume_ = world_top_devptr.GetPtr();
    CELER_CUDA_CHECK_ERROR();
    print_time(get_time());

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
GeoParamsPointers GeoParams::host_pointers() const
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
 * For the cms2018.gdml detector geometry, the default stack size is too small,
 * and a limit of 32768 is recommended.
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
