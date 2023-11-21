//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "orange/BoundingBox.hh"
#include "orange/GeoParamsInterface.hh"
#include "orange/Types.hh"

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

    // Construct from a GDML filename
    explicit VecgeomParams(std::string const& gdml_filename);

    // Create a VecGeom model from a pre-existing Geant4 geometry
    explicit VecgeomParams(G4VPhysicalVolume const* world);

    // Clean up VecGeom on destruction
    ~VecgeomParams();

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return true; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    //! Maximum nested geometry depth
    int max_depth() const { return host_ref_.max_depth; }

    //// VOLUMES ////

    //! Number of volumes
    VolumeId::size_type num_volumes() const final
    {
        return vol_labels_.size();
    }

    // Get the label for a placed volume ID
    Label const& id_to_label(VolumeId vol_id) const final;

    //! \cond
    using GeoParamsInterface::find_volume;
    //! \endcond

    // Get the volume ID corresponding to a unique label name
    VolumeId find_volume(std::string const& name) const final;

    // Get the volume ID corresponding to a unique label
    VolumeId find_volume(Label const& label) const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    VolumeId find_volume(G4LogicalVolume const* volume) const final;

    // Get zero or more volume IDs corresponding to a name
    SpanConstVolumeId find_volumes(std::string const& name) const final;

    //// DATA ACCESS ////

    //! Access geometry data on host
    HostRef const& host_ref() const final { return host_ref_; }

    //! Access geometry data on device
    DeviceRef const& device_ref() const final { return device_ref_; }

  private:
    //// DATA ////

    // Host metadata/access
    LabelIdMultiMap<VolumeId> vol_labels_;
    std::unordered_map<G4LogicalVolume const*, VolumeId> g4log_volid_map_;

    BBox bbox_;

    // Host/device storage and reference
    HostRef host_ref_;
    DeviceRef device_ref_;

    //// HELPER FUNCTIONS ////

    // Construct VecGeom tracking data and copy to GPU
    void build_tracking();
    void build_surface_tracking();
    void build_volume_tracking();

    // Construct host/device Celeritas data
    void build_data();
    // Construct labels and other host-only metadata
    void build_metadata();
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
