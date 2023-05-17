//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Label.hh"

#include "BoundingBox.hh"
#include "GeoParamsInterface.hh"
#include "OrangeData.hh"
#include "OrangeTypes.hh"

class G4VPhysicalVolume;

namespace celeritas
{
struct OrangeInput;

//---------------------------------------------------------------------------//
/*!
 * Persistent model data for an ORANGE geometry.
 *
 * This class initializes and manages the data used by ORANGE (surfaces,
 * volumes) and provides a host-based interface for them.
 */
class OrangeParams final : public GeoParamsSurfaceInterface,
                           public ParamsDataInterface<OrangeParamsData>
{
  public:
    // Construct from a JSON file (if JSON is enabled)
    explicit OrangeParams(std::string const& json_filename);

    // Construct in-memory from Geant4 (not implemented)
    explicit OrangeParams(G4VPhysicalVolume const*);

    // ADVANCED usage: construct from explicit host data
    explicit OrangeParams(OrangeInput input);

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return supports_safety_; }

    //! Outer bounding box of geometry
    BoundingBox const& bbox() const final { return bbox_; }

    //// VOLUMES ////

    // Number of volumes
    inline VolumeId::size_type num_volumes() const final;

    // Get the label for a placed volume ID
    Label const& id_to_label(VolumeId vol_id) const final;

    //! \cond
    using GeoParamsSurfaceInterface::find_volume;
    //! \endcond

    // Get the volume ID corresponding to a unique name
    VolumeId find_volume(std::string const& name) const final;

    // Get the volume ID corresponding to a unique label
    VolumeId find_volume(Label const& label) const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    inline VolumeId find_volume(G4LogicalVolume const* volume) const final;

    // Get zero or more volume IDs corresponding to a name
    SpanConstVolumeId find_volumes(std::string const& name) const final;

    //// SURFACES ////

    // Get the label for a placed volume ID
    Label const& id_to_label(SurfaceId surf_id) const final;

    // Get the surface ID corresponding to a unique label name
    SurfaceId find_surface(std::string const& name) const final;

    //! Number of distinct surfaces
    inline SurfaceId::size_type num_surfaces() const final;

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    HostRef const& host_ref() const final { return data_.host(); }

    //! Reference to managed GPU geometry data
    DeviceRef const& device_ref() const final { return data_.device(); }

  private:
    // Host metadata/access
    LabelIdMultiMap<SurfaceId> surf_labels_;
    LabelIdMultiMap<VolumeId> vol_labels_;
    BoundingBox bbox_;
    bool supports_safety_{};

    // Host/device storage and reference
    CollectionMirror<OrangeParamsData> data_;

  private:
    //// HELPER METHODS ////

    // Get surface and volume labels for all universes.
    void process_metadata(OrangeInput const& input);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 volume.
 *
 * TODO: To be properly implemented, as it requires a future Geant4 converter.
 */
VolumeId OrangeParams::find_volume(G4LogicalVolume const*) const
{
    return VolumeId{};
}

//---------------------------------------------------------------------------//
/*!
 * Number of volumes.
 */
VolumeId::size_type OrangeParams::num_volumes() const
{
    return vol_labels_.size();
}

//---------------------------------------------------------------------------//
/*!
 * Number of distinct surfaces.
 */
SurfaceId::size_type OrangeParams::num_surfaces() const
{
    return surf_labels_.size();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
