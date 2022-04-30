//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecgeomParams.cc
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include <regex>
#include <set>
#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include <celeritas_config.h>
#if CELERITAS_USE_CUDA
#    include <VecGeom/management/CudaManager.h>
#    include <cuda_runtime_api.h>
#endif

#include "base/Join.hh"
#include "base/Range.hh"
#include "base/ScopedTimeAndRedirect.hh"
#include "base/StringUtils.hh"
#include "comm/Device.hh"
#include "comm/Logger.hh"

#include "VecgeomData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
VecgeomParams::VecgeomParams(const std::string& filename)
{
    CELER_LOG(info) << "Loading Geant4 geometry from GDML at " << filename;
    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    {
        ScopedTimeAndRedirect time_and_output_("vgdml::Frontend");
        constexpr bool        validate_xml_schema = false;
        vgdml::Frontend::Load(filename, validate_xml_schema);
    }

    CELER_LOG(status) << "Initializing tracking information";
    {
        ScopedTimeAndRedirect time_and_output_("vecgeom::ABBoxManager");
        vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    }

    // Create metadata
    this->build_md();

    // Save host data
    auto& vg_manager       = vecgeom::GeoManager::Instance();
    host_ref_.world_volume = vg_manager.GetWorld();
    host_ref_.max_depth    = vg_manager.getMaxDepth();

    // Init the bounding volume hierarchy structure
    vecgeom::cxx::BVHManager::Init();

#if CELERITAS_USE_CUDA
    if (celeritas::device())
    {
        auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();

        CELER_LOG(debug) << "Converting to CUDA geometry";
        {
            ScopedTimeAndRedirect time_and_output_("vecgeom::CudaManager");
            // cuda_manager.set_verbose(1);
            cuda_manager.LoadGeometry();
            CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
        }

        CELER_LOG(debug) << "Transferring geometry to GPU";
        {
            ScopedTimeAndRedirect time_and_output_("vecgeom::CudaManager");
            auto world_top_devptr = cuda_manager.Synchronize();
            CELER_ASSERT(world_top_devptr != nullptr);
            device_ref_.world_volume = world_top_devptr.GetPtr();
            device_ref_.max_depth    = host_ref_.max_depth;
            CELER_DEVICE_CHECK_ERROR();
        }
        CELER_ENSURE(device_ref_);

        CELER_LOG(debug) << "Initailizing BVH on GPU";
        {
            vecgeom::cxx::BVHManager::DeviceInit();
            CELER_DEVICE_CHECK_ERROR();
        }
    }
#endif

    CELER_ENSURE(this->num_volumes() > 0);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up vecgeom on destruction.
 */
VecgeomParams::~VecgeomParams()
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
 * Get the label for a placed volume ID.
 */
const std::string& VecgeomParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_[vol.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label.
 */
auto VecgeomParams::find_volume(const std::string& label) const -> VolumeId
{
    auto iter = vol_ids_.find(label);
    if (iter == vol_ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Construct label metadata from volumes.
 */
void VecgeomParams::build_md()
{
    auto& vg_manager = vecgeom::GeoManager::Instance();

    vol_labels_.resize(vg_manager.GetRegisteredVolumesCount());
    std::set<std::string> duplicate_volumes;

    const std::regex final_ptr_regex{"0x[0-9a-f]{8,16}$"};
    std::smatch      ptr_match;

    for (auto vol_idx : range<VolumeId::size_type>(vol_labels_.size()))
    {
        // Get label
        const vecgeom::LogicalVolume* vol
            = vg_manager.FindLogicalVolume(vol_idx);
        CELER_ASSERT(vol);
        std::string vol_label = vol->GetLabel();

        // Remove possible Geant uniquifying pointer-address suffix
        // (Geant4 does this automatically, but VGDML does not)
        if (std::regex_search(vol_label, ptr_match, final_ptr_regex))
        {
            vol_label.erase(vol_label.begin() + ptr_match.position(0),
                            vol_label.end());
        }

        // Add to label-to-ID map
        auto iter_inserted = vol_ids_.insert({vol_label, VolumeId{vol_idx}});
        if (!iter_inserted.second)
        {
            duplicate_volumes.insert(vol_label);
        }

        // Move to volume label
        vol_labels_[vol_idx] = std::move(vol_label);
    }

    if (!duplicate_volumes.empty())
    {
        CELER_LOG(warning)
            << "Geometry contains duplicate volume names: "
            << join(duplicate_volumes.begin(), duplicate_volumes.end(), ", ");
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
