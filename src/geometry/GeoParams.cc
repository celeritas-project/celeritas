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
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <celeritas_config.h>
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#    include <VecGeom/management/CudaManager.h>
#endif

#include "base/StringUtils.hh"
#include "comm/Device.hh"
#include "comm/Logger.hh"
#include "GeoData.hh"
#include "detail/ScopedTimeAndRedirect.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
GeoParams::GeoParams(const std::string& filename)
{
    CELER_LOG(info) << "Loading Geant4 geometry from GDML at " << filename;
    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    {
        detail::ScopedTimeAndRedirect time_and_output_;
        constexpr bool                validate_xml_schema = false;
        vgdml::Frontend::Load(filename, validate_xml_schema);
    }

    CELER_LOG(status) << "Initializing tracking information";
    {
        detail::ScopedTimeAndRedirect time_and_output_;
        vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    }

    num_volumes_ = vecgeom::GeoManager::Instance().GetRegisteredVolumesCount();

    host_ref_.world_volume = vecgeom::GeoManager::Instance().GetWorld();
    host_ref_.max_depth    = vecgeom::GeoManager::Instance().getMaxDepth();

    // init the BVH structure
    vecgeom::cxx::BVHManager::Init();

#if CELERITAS_USE_CUDA
    if (celeritas::device())
    {
        auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();

        CELER_LOG(status) << "Converting to CUDA geometry";
        {
            detail::ScopedTimeAndRedirect time_and_output_;
            // cuda_manager.set_verbose(1);
            cuda_manager.LoadGeometry();
            CELER_CUDA_CALL(cudaDeviceSynchronize());
        }

        CELER_LOG(status) << "Transferring geometry to GPU";
        {
            detail::ScopedTimeAndRedirect time_and_output_;
            auto world_top_devptr = cuda_manager.Synchronize();
            CELER_ASSERT(world_top_devptr != nullptr);
            device_ref_.world_volume = world_top_devptr.GetPtr();
            device_ref_.max_depth    = host_ref_.max_depth;
            CELER_CUDA_CHECK_ERROR();
        }
        CELER_ENSURE(device_ref_);

        // init the BVH structure
        vecgeom::cxx::BVHManager::DeviceInit();
    }
#endif

    CELER_ENSURE(num_volumes_ > 0);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up vecgeom on destruction
 */
GeoParams::~GeoParams()
{
#if CELERITAS_USE_CUDA
    if (device_ref_)
    {
        CELER_LOG(debug) << "Clearing VecGeom GPU data";
        vecgeom::CudaManager::Instance().Clear();
    }
#endif
    CELER_LOG(debug) << "Clearing VecGeom CPU data";
    vecgeom::GeoManager::Instance().Clear();
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a placed volume ID
 */
const std::string& GeoParams::id_to_label(VolumeId vol_id) const
{
    CELER_EXPECT(vol_id.get() < num_volumes_);
    const auto* vol
        = vecgeom::GeoManager::Instance().FindLogicalVolume(vol_id.get());
    CELER_ASSERT(vol);
    return vol->GetLabel();
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label
 */
auto GeoParams::label_to_id(const std::string& label) const -> VolumeId
{
    const auto* vol
        = vecgeom::GeoManager::Instance().FindLogicalVolume(label.c_str());
    CELER_ASSERT(vol);
    CELER_ASSERT(vol->id() < num_volumes_);
    return VolumeId{vol->id()};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
