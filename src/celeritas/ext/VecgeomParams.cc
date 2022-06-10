//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomParams.cc
//---------------------------------------------------------------------------//
#include "VecgeomParams.hh"

#include <vector>
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Cuda.h>
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

#include "corecel/cont/Range.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/ext/detail/G4VecgeomConverter.hh"

#include "VecgeomData.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Get a vector of labels for all vecgeom volumes.
std::vector<Label> get_volume_labels()
{
    auto& vg_manager = vecgeom::GeoManager::Instance();

    std::vector<Label> labels(vg_manager.GetRegisteredVolumesCount());

    for (auto vol_idx : range<VolumeId::size_type>(labels.size()))
    {
        // Get label
        const vecgeom::LogicalVolume* vol
            = vg_manager.FindLogicalVolume(vol_idx);
        CELER_ASSERT(vol);

        auto label = Label::from_geant(vol->GetLabel());
        if (label.name.empty())
        {
            // Many VGDML imported IDs seem to be empty for CMS
            label.name = "[unused]";
            label.ext  = std::to_string(vol_idx);
        }

        labels[vol_idx] = std::move(label);
    }
    return labels;
}
//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
VecgeomParams::VecgeomParams(const std::string& filename)
{
    CELER_LOG(info) << "Loading VecGeom geometry from GDML at " << filename;
    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    {
        ScopedTimeAndRedirect time_and_output_("vgdml::Frontend");
        constexpr bool        validate_xml_schema = false;
        vgdml::Frontend::Load(filename, validate_xml_schema);
    }

    vol_labels_ = LabelIdMultiMap<VolumeId>(get_volume_labels());
    // Check for duplicates
    {
        auto vol_dupes = vol_labels_.duplicates();
        if (!vol_dupes.empty())
        {
            auto streamed_label = [this](std::ostream& os, VolumeId v) {
                os << '"' << this->vol_labels_.get(v) << "\" ("
                   << v.unchecked_get() << ')';
            };

            CELER_LOG(warning) << "Geometry contains duplicate volume names: "
                               << join_stream(vol_dupes.begin(),
                                              vol_dupes.end(),
                                              ", ",
                                              streamed_label);
        }
    }

    Initialize();
}

#ifdef CELERITAS_USE_GEANT4
//---------------------------------------------------------------------------//
/*!
 *  Create a VecGeom model from an pre-existing Geant4 geometry
 */
VecgeomParams::VecgeomParams(const G4VPhysicalVolume* p_G4world)
{
    CELER_LOG(info) << "Creating VecGeom model from pre-existing G4 geometry";
    CELER_ASSERT(p_G4world);

    // Convert the geometry to VecGeom
    G4VecGeomConverter::Instance().SetVerbose(1);
    G4VecGeomConverter::Instance().ConvertG4Geometry(p_G4world);
    CELER_LOG(info) << "Converted: max_depth = "
                    << vecgeom::GeoManager::Instance().getMaxDepth();

    //.. dump VecGeom geometry details for comparison
    vecgeom::VPlacedVolume const* vgWorld
        = vecgeom::GeoManager::Instance().GetWorld();
    CELER_ENSURE(vgWorld);

    Initialize();
}
#endif

void VecgeomParams::Initialize()
{
    CELER_LOG(status) << "Initializing tracking information";
    {
        ScopedTimeAndRedirect time_and_output_("vecgeom::ABBoxManager");
        vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    }

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

        CELER_LOG(debug) << "Initializing BVH on GPU";
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
const Label& VecgeomParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_.get(vol);
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label.
 */
auto VecgeomParams::find_volume(const std::string& name) const -> VolumeId
{
    auto result = vol_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "volume '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
/*!
 * Get zero or more volume IDs corresponding to a name.
 *
 * This is useful for volumes that are repeated in the geometry with different
 * uniquifying 'extensions' from Geant4.
 */
auto VecgeomParams::find_volumes(const std::string& name) const
    -> SpanConstVolumeId
{
    return vol_labels_.find_all(name);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
