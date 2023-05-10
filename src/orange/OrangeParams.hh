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

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/io/Label.hh"

#include "BoundingBox.hh"
#include "OrangeData.hh"
#include "OrangeTypes.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;

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
class OrangeParams
{
  public:
    //!@{
    //! \name Type aliases
    using HostRef = HostCRef<OrangeParamsData>;
    using DeviceRef = DeviceCRef<OrangeParamsData>;
    using SpanConstVolumeId = Span<VolumeId const>;
    //!@}

  public:
    // Construct from a JSON file (if JSON is enabled)
    explicit OrangeParams(std::string const& json_filename);

    // Construct in-memory from Geant4 (not implemented)
    explicit OrangeParams(G4VPhysicalVolume const*);

    // ADVANCED usage: construct from explicit host data
    explicit OrangeParams(OrangeInput input);

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const { return supports_safety_; }

    //! Outer bounding box of geometry
    BoundingBox const& bbox() const { return bbox_; }

    //// VOLUMES ////

    //! Number of volumes
    VolumeId::size_type num_volumes() const { return vol_labels_.size(); }

    // Get the label for a placed volume ID
    Label const& id_to_label(VolumeId vol_id) const;

    // Get the volume ID corresponding to a unique name
    inline VolumeId find_volume(char const* name) const;

    // Get the volume ID corresponding to a unique name
    VolumeId find_volume(std::string const& name) const;

    // Get the volume ID corresponding to a unique label
    VolumeId find_volume(Label const& label) const;

    // Get the volume ID corresponding to a Geant4 logical volume
    VolumeId find_volume(G4LogicalVolume const* volume) const;

    // Get zero or more volume IDs corresponding to a name
    SpanConstVolumeId find_volumes(std::string const& name) const;

    //// SURFACES ////

    // Get the label for a placed volume ID
    Label const& id_to_label(SurfaceId surf_id) const;

    // Get the surface ID corresponding to a unique label name
    SurfaceId find_surface(std::string const& name) const;

    //! Number of distinct surfaces
    SurfaceId::size_type num_surfaces() const { return surf_labels_.size(); }

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    HostRef const& host_ref() const { return data_.host(); }

    //! Reference to managed GPU geometry data
    DeviceRef const& device_ref() const { return data_.device(); }

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
 * Find the unique volume corresponding to a unique name.
 *
 * This method is here to disambiguate the implicit std::string and Label
 * constructors.
 */
VolumeId OrangeParams::find_volume(char const* name) const
{
    return this->find_volume(std::string{name});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
