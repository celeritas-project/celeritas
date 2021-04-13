//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoParams.cc
//---------------------------------------------------------------------------//
#include "GeoParams.hh"

#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <celeritas_config.h>
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#    include <VecGeom/management/CudaManager.h>
#endif

#include "comm/Device.hh"
#include "comm/Logger.hh"
#include "GeoInterface.hh"
#include "detail/ScopedTimeAndRedirect.hh"

#if CELERITAS_USE_ROOT && defined(VECGEOM_ROOT)
#    include "TGeoManager.h"
#    include "VecGeom/management/RootGeoManager.h"
#else
#    include <VecGeom/gdml/Frontend.h>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
GeoParams::GeoParams(const char* gdml_filename)
{
    {
        detail::ScopedTimeAndRedirect time_and_output_;

#if CELERITAS_USE_ROOT && defined(VECGEOM_ROOT)
        // use Root available from VecGeom
        CELER_LOG(info) << "RootGeoManager parsing: Loading from GDML at "
                        << gdml_filename;
        vecgeom::RootGeoManager::Instance().set_verbose(1);
        vecgeom::RootGeoManager::Instance().LoadRootGeometry(gdml_filename);
#else
        CELER_LOG(info) << "VecGeom parsing: Loading from GDML at "
                        << gdml_filename;
        constexpr bool validate_xml_schema = false;
	real_type mm_scale = 1.0;    // indirectly sets default VecGeom units
        vgdml::Frontend::Load(gdml_filename, validate_xml_schema, mm_scale);
#endif
    }

    CELER_LOG(status) << "Initializing tracking information";
    {
        detail::ScopedTimeAndRedirect time_and_output_;
        vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    }

    num_volumes_ = vecgeom::VPlacedVolume::GetIdCount();
    max_depth_   = vecgeom::GeoManager::Instance().getMaxDepth();

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
            device_world_volume_ = world_top_devptr.GetPtr();
            CELER_CUDA_CHECK_ERROR();
        }

        CELER_ENSURE(device_world_volume_);
    }
#endif

    CELER_ENSURE(num_volumes_ > 0);
    CELER_ENSURE(max_depth_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up vecgeom on destruction
 */
GeoParams::~GeoParams()
{
#if CELERITAS_USE_CUDA
    if (device_world_volume_)
    {
        // NOTE: if the following line fails to compile, you need to update
        // VecGeom to at least 1.1.12 (or after 2021FEB17)
        vecgeom::CudaManager::Instance().Clear();
    }
#endif
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
        = vecgeom::GeoManager::Instance().FindPlacedVolume(vol_id.get());
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
        = vecgeom::GeoManager::Instance().FindPlacedVolume(label.c_str());
    CELER_ASSERT(vol);
    CELER_ASSERT(vol->id() < num_volumes_);
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
    CELER_EXPECT(celeritas::device());
    GeoParamsPointers result;
    result.world_volume
        = static_cast<const vecgeom::VPlacedVolume*>(device_world_volume_);
    CELER_ENSURE(result);
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
    CELER_EXPECT(limit > 0);
    CELER_EXPECT(celeritas::device());
    CELER_CUDA_CALL(cudaDeviceSetLimit(cudaLimitStackSize, limit));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
