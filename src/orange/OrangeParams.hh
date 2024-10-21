//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "geocel/BoundingBox.hh"
#include "geocel/GeoParamsInterface.hh"

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
    //!@{
    //! \name Type aliases
    using SurfaceMap = LabelIdMultiMap<SurfaceId>;
    using UniverseMap = LabelIdMultiMap<UniverseId>;
    //!@}

  public:
    // Construct from a JSON or GDML file (if JSON or Geant4 are enabled)
    explicit OrangeParams(std::string const& filename);

    // Construct in-memory from Geant4
    explicit OrangeParams(G4VPhysicalVolume const*);

    // ADVANCED usage: construct from explicit host data
    explicit OrangeParams(OrangeInput&& input);

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return supports_safety_; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    //! Maximum universe depth
    size_type max_depth() const { return this->host_ref().scalars.max_depth; }

    //// LABELS AND MAPPING ////

    // Get surface metadata
    inline SurfaceMap const& surfaces() const final;

    // Get universe metadata
    inline UniverseMap const& universes() const;

    // Get volume metadata
    inline VolumeMap const& volumes() const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    inline VolumeId find_volume(G4LogicalVolume const* volume) const final;

    //// DEPRECATED ////

    using GeoParamsSurfaceInterface::find_volume;
    using GeoParamsSurfaceInterface::id_to_label;

    // Get the label for a universe ID
    [[deprecated]]
    Label const& id_to_label(UniverseId univ_id) const
    {
        return this->universes().at(univ_id);
    }

    // Get the universe ID corresponding to a unique label name
    [[deprecated]]
    UniverseId find_universe(std::string const& name) const
    {
        return this->universes().find_unique(name);
    }

    // Number of universes
    [[deprecated]]
    UniverseId::size_type num_universes() const
    {
        return this->universes().size();
    }

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Reference to managed GPU geometry data
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host metadata/access
    SurfaceMap surf_labels_;
    UniverseMap univ_labels_;
    VolumeMap vol_labels_;
    BBox bbox_;
    bool supports_safety_{};

    // Host/device storage and reference
    CollectionMirror<OrangeParamsData> data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get surface metadata.
 */
auto OrangeParams::surfaces() const -> SurfaceMap const&
{
    return surf_labels_;
}

//---------------------------------------------------------------------------//
/*!
 * Get universe metadata.
 */
auto OrangeParams::universes() const -> UniverseMap const&
{
    return univ_labels_;
}

//---------------------------------------------------------------------------//
/*!
 * Get volume metadata.
 */
auto OrangeParams::volumes() const -> VolumeMap const&
{
    return vol_labels_;
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 volume.
 *
 * \todo Implement using \c g4org::Converter
 */
VolumeId OrangeParams::find_volume(G4LogicalVolume const*) const
{
    return VolumeId{};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
