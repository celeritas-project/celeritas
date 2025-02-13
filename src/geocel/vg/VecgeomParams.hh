//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/Types.hh"

#include "VecgeomData.hh"

class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared model parameters for a VecGeom geometry.
 *
 * The model defines the shapes, volumes, etc.
 */
class VecgeomParams final : public GeoParamsInterface,
                            public ParamsDataInterface<VecgeomParamsData>
{
  public:
    // Whether surface tracking is being used
    static bool use_surface_tracking();

    // Whether VecGeom GDML is being used to load the geometry
    static bool use_vgdml();

    // Construct from a GDML filename
    explicit VecgeomParams(std::string const& gdml_filename);

    // Create a VecGeom model from a pre-existing Geant4 geometry
    explicit VecgeomParams(G4VPhysicalVolume const* world);

    // Clean up VecGeom on destruction
    ~VecgeomParams() final;

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return true; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    //! Maximum nested geometry depth
    LevelId::size_type max_depth() const final { return host_ref_.max_depth; }

    //// VOLUMES ////

    // Get volume metadata
    inline VolumeMap const& volumes() const final;

    // Get (physical) volume instance metadata
    inline VolInstanceMap const& volume_instances() const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    VolumeId find_volume(G4LogicalVolume const* volume) const final;

    // Get the Geant4 physical volume corresponding to a volume instance ID
    G4VPhysicalVolume const* id_to_pv(VolumeInstanceId vol_id) const final;

    // DEPRECATED
    using GeoParamsInterface::find_volume;

    //// DATA ACCESS ////

    //! Access geometry data on host
    HostRef const& host_ref() const final { return host_ref_; }

    //! Access geometry data on device
    DeviceRef const& device_ref() const final { return device_ref_; }

  private:
    //// DATA ////

    // Host metadata/access
    LabelIdMultiMap<VolumeId> volumes_;
    VolInstanceMap vol_instances_;
    std::unordered_map<G4LogicalVolume const*, VolumeId> g4log_volid_map_;
    std::vector<G4VPhysicalVolume const*> g4_pv_map_;

    BBox bbox_;

    // Host/device storage and reference
    HostRef host_ref_;
    DeviceRef device_ref_;

    // If VGDML is unavailable and Geant4 is, we load and
    bool loaded_geant4_gdml_{false};

    //// HELPER FUNCTIONS ////

    // Construct VecGeom tracking data and copy to GPU
    void build_volumes_vgdml(std::string const& filename);
    void build_volumes_geant4(G4VPhysicalVolume const* world);
    void build_tracking();
    void build_surface_tracking();
    void build_volume_tracking();

    // Construct host/device Celeritas data
    void build_data();
    // Construct labels and other host-only metadata
    void build_metadata();
};

//---------------------------------------------------------------------------//
/*!
 * Get volume metadata.
 */
auto VecgeomParams::volumes() const -> VolumeMap const&
{
    return volumes_;
}

//---------------------------------------------------------------------------//
/*!
 * Get volume instance metadata.
 *
 * Volume instances correspond directly to Geant4 physical volumes.
 */
auto VecgeomParams::volume_instances() const -> VolInstanceMap const&
{
    return vol_instances_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
