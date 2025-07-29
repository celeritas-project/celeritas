//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Label.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/inp/Model.hh"

#include "OrangeData.hh"
#include "OrangeTypes.hh"

class G4VPhysicalVolume;

namespace celeritas
{
struct OrangeInput;
class GeantGeoParams;

//---------------------------------------------------------------------------//
/*!
 * Persistent model data for an ORANGE geometry.
 *
 * This class initializes and manages the data used by ORANGE (surfaces,
 * volumes) and provides a host-based interface for them.
 */
class OrangeParams final : public GeoParamsInterface,
                           public ParamsDataInterface<OrangeParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceMap = LabelIdMultiMap<ImplSurfaceId>;
    using UniverseMap = LabelIdMultiMap<UniverseId>;
    //!@}

  public:
    //!@{
    //! \name Static constructor helpers
    //! \todo: move these to a "model" abstraction that loads/emits geometry,
    //! materials, volumes?

    // Build by loading a GDML file
    static std::shared_ptr<OrangeParams> from_gdml(std::string const& filename);

    // Build from a Geant4 geometry
    static std::shared_ptr<OrangeParams>
    from_geant(std::shared_ptr<GeantGeoParams const> const& geo);

    // Build from a JSON input
    static std::shared_ptr<OrangeParams> from_json(std::string const& filename);

    //!@}

    // ADVANCED usage: construct from explicit host data
    explicit OrangeParams(OrangeInput&& input);

    // Default destructor to anchor vtable
    ~OrangeParams() final;

    // Moving would leave the class in an unspecified state
    CELER_DELETE_COPY_MOVE(OrangeParams);

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return supports_safety_; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    // Maximum universe depth (not geometry volume depth!)
    inline size_type max_depth() const;

    // Create model parameters corresponding to our internal representation
    inp::Model make_model_input() const final;

    //// LABELS AND MAPPING ////

    // Get surface metadata
    inline SurfaceMap const& surfaces() const;

    // Get universe metadata
    inline UniverseMap const& universes() const;

    // Get volume metadata
    inline ImplVolumeMap const& impl_volumes() const final;

    // Get (physical) volume instance metadata
    inline VolInstanceMap const& volume_instances() const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    inline ImplVolumeId find_volume(G4LogicalVolume const* volume) const final;

    // Get the Geant4 physical volume corresponding to a volume instance ID
    inline GeantPhysicalInstance
    id_to_geant(VolumeInstanceId vol_id) const final;

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Reference to managed GPU geometry data
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host metadata/access
    SurfaceMap surf_labels_;
    UniverseMap univ_labels_;
    ImplVolumeMap vol_labels_;
    VolInstanceMap vol_instances_;
    BBox bbox_;
    bool supports_safety_{};

    // Host/device storage and reference
    CollectionMirror<OrangeParamsData> data_;
};

//---------------------------------------------------------------------------//

extern template class CollectionMirror<OrangeParamsData>;
extern template class ParamsDataInterface<OrangeParamsData>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Maximum universe depth.
 */
size_type OrangeParams::max_depth() const
{
    return this->host_ref().scalars.max_depth;
}

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
auto OrangeParams::impl_volumes() const -> ImplVolumeMap const&
{
    return vol_labels_;
}

//---------------------------------------------------------------------------//
/*!
 * Get volume instance metadata.
 */
auto OrangeParams::volume_instances() const -> VolInstanceMap const&
{
    return vol_instances_;
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 volume.
 *
 * \todo Implement using \c g4org::Converter
 */
ImplVolumeId OrangeParams::find_volume(G4LogicalVolume const*) const
{
    return ImplVolumeId{};
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 physical volume corresponding to a volume instance ID.
 *
 * \todo Implement using \c g4org::Converter
 */
GeantPhysicalInstance OrangeParams::id_to_geant(VolumeInstanceId) const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
