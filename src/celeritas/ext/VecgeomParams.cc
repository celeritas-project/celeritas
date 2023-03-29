//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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

#include "celeritas_config.h"
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

#include "VecgeomData.hh"  // IWYU pragma: associated
#include "detail/GeantGeoExporter.hh"
#include "detail/GeantGeometryImport.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
VecgeomParams::VecgeomParams(std::string const& filename)
{
    CELER_LOG(status) << "Loading VecGeom geometry from GDML at " << filename;
    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    {
        ScopedTimeAndRedirect time_and_output_("vgdml::Frontend");
        vgdml::Frontend::Load(filename, /* validate_xml_schema = */ false);
    }

    this->build_tracking();
    this->build_data();
    this->build_metadata();

    CELER_ENSURE(this->num_volumes() > 0);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Translate a geometry from Geant4.
 */
VecgeomParams::VecgeomParams(G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    CELER_LOG(status) << "Importing VecGeom model from Geant4";

    // Convert the geometry to VecGeom
    detail::g4_to_vecgeom(world);
    CELER_LOG(debug) << "Converted: max_depth = "
                     << vecgeom::GeoManager::Instance().getMaxDepth();

    //.. dump VecGeom geometry details for comparison
    vecgeom::VPlacedVolume const* vgWorld
        = vecgeom::GeoManager::Instance().GetWorld();
    CELER_ASSERT(vgWorld);
    CELER_LOG(debug) << "Top VecGeom volume: " << vgWorld->GetName()
                     << " - Label: " << vgWorld->GetLabel();

    this->build_tracking();
    this->build_data();
    this->build_metadata();

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
Label const& VecgeomParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_.get(vol);
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label.
 */
auto VecgeomParams::find_volume(std::string const& name) const -> VolumeId
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
 * Locate the volume ID corresponding to a label.
 *
 * If the label isn't in the geometry, a null ID will be returned.
 */
VolumeId VecgeomParams::find_volume(Label const& label) const
{
    return vol_labels_.find(label);
}

//---------------------------------------------------------------------------//
/*!
 * Get zero or more volume IDs corresponding to a name.
 *
 * This is useful for volumes that are repeated in the geometry with different
 * uniquifying 'extensions' from Geant4.
 */
auto VecgeomParams::find_volumes(std::string const& name) const
    -> SpanConstVolumeId
{
    return vol_labels_.find_all(name);
}

//---------------------------------------------------------------------------//
/*!
 * After loading solids, set up VecGeom tracking data and copy to GPU.
 */
void VecgeomParams::build_tracking()
{
    CELER_LOG(status) << "Initializing tracking information";
    {
        ScopedTimeAndRedirect time_and_output_("vecgeom::ABBoxManager");
        vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
    }

    // Init the bounding volume hierarchy structure
    vecgeom::cxx::BVHManager::Init();

    if (celeritas::device())
    {
        // NOTE: this must actually be escaped with preprocessing because the
        // VecGeom interfaces change depending on the build options.
#if CELERITAS_USE_CUDA
        auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();

        // Insure that kernels, for which the compiler is not able to determine
        // the total stack usage, have enough stack reserved.
        // See https://github.com/celeritas-project/celeritas/issues/614 for
        // a more detailed discussion.
        // Experimentally, this seems to be needed only in the following cases.
        // (a stricter condition could have been
        //   if use_vecgeom && (compiled without -O2 || CELERITAS_DEBUG)
        //      || (CELERITAS_DEBUG && compiled with -g)
        // with the 2nd part being usefull only if vulnerable kernels start
        // appearing in the build; for this second part we ought to move (or
        // more exactly copy) to `celeritas::activate_device` as the latest
        // cudaDeviceSetLimit 'wins'.
        if constexpr (CELERITAS_DEBUG)
        {
            set_cuda_stack_size(16384);
        }

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
            CELER_DEVICE_CHECK_ERROR();
        }

        CELER_LOG(debug) << "Initializing BVH on GPU";
        {
            vecgeom::cxx::BVHManager::DeviceInit();
            CELER_DEVICE_CHECK_ERROR();
        }
#else
        CELER_NOT_CONFIGURED("CUDA");
#endif
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct host/device Celeritas data after setting up VecGeom tracking.
 */
void VecgeomParams::build_data()
{
    // Save host data
    auto& vg_manager = vecgeom::GeoManager::Instance();
    host_ref_.world_volume = vg_manager.GetWorld();
    host_ref_.max_depth = vg_manager.getMaxDepth();

    if (celeritas::device())
    {
#if CELERITAS_USE_CUDA
        auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
        device_ref_.world_volume = cuda_manager.world_gpu();
#endif
        device_ref_.max_depth = host_ref_.max_depth;
    }
    CELER_ENSURE(host_ref_);
    CELER_ENSURE(!celeritas::device() || device_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas host-only metadata.
 */
void VecgeomParams::build_metadata()
{
    auto& vg_manager = vecgeom::GeoManager::Instance();
    CELER_EXPECT(vg_manager.GetRegisteredVolumesCount() > 0);

    std::vector<Label> labels(vg_manager.GetRegisteredVolumesCount());

    for (auto vol_idx : range<VolumeId::size_type>(labels.size()))
    {
        // Get label
        vecgeom::LogicalVolume const* vol
            = vg_manager.FindLogicalVolume(vol_idx);
        CELER_ASSERT(vol);

        auto label = Label::from_geant(vol->GetLabel());
        if (label.name.empty())
        {
            // Many VGDML imported IDs seem to be empty for CMS
            label.name = "[unused]";
            label.ext = std::to_string(vol_idx);
        }

        labels[vol_idx] = std::move(label);
    }
    vol_labels_ = LabelIdMultiMap<VolumeId>(std::move(labels));
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
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
