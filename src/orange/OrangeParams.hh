//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
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
class VolumeParams;

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
    using UniverseMap = LabelIdMultiMap<UnivId>;
    using SPConstVolumes = std::shared_ptr<VolumeParams const>;
    //!@}

  public:
    //!@{
    //! \name Static constructor helpers
    //! \todo Move these to a "model" abstraction that loads/emits geometry,
    //! materials, volumes?

    // Build by loading a GDML file
    static std::shared_ptr<OrangeParams> from_gdml(std::string const& filename);

    // Build from a Geant4 geometry
    static std::shared_ptr<OrangeParams>
    from_geant(std::shared_ptr<GeantGeoParams const> const& geo,
               SPConstVolumes volumes);

    // Build from a Geant4 geometry (no volumes available?)
    static std::shared_ptr<OrangeParams>
    from_geant(std::shared_ptr<GeantGeoParams const> const& geo);

    // Build from a JSON input
    static std::shared_ptr<OrangeParams> from_json(std::string const& filename);

    //!@}

    // ADVANCED usage: construct from explicit host data
    OrangeParams(OrangeInput&& input);

    // ADVANCED usage: construct from explicit host data with volumes
    OrangeParams(OrangeInput&& input, SPConstVolumes&& volumes);

    // Default destructor to anchor vtable
    ~OrangeParams() final;

    // Moving would leave the class in an unspecified state
    CELER_DELETE_COPY_MOVE(OrangeParams);

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return supports_safety_; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    // Number of levels deep universes can be (not geometry volume level!)
    inline UnivLevelId::size_type num_univ_levels() const;

    // Create model parameters corresponding to our internal representation
    inp::Model make_model_input() const final;

    //// LABELS AND MAPPING ////

    // Get surface metadata
    inline SurfaceMap const& surfaces() const;

    // Get universe metadata
    inline UniverseMap const& universes() const;

    // Get volume metadata
    inline ImplVolumeMap const& impl_volumes() const final;

    // Get the canonical volume IDs corresponding to an implementation volume
    inline VolumeId volume_id(ImplVolumeId) const final;

    // Get the (canonical) volume instance ID corresponding to an
    // implementation volume
    inline VolumeInstanceId volume_instance_id(ImplVolumeId) const;

    // Get the volume instance containing the global point
    VolumeInstanceId find_volume_instance_at(Real3 const&) const final;

    //// DATA ACCESS ////

    //! Reference to CPU geometry data
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Reference to managed GPU geometry data
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host metadata/access
    SurfaceMap impl_surf_labels_;
    UniverseMap univ_labels_;
    ImplVolumeMap impl_vol_labels_;
    BBox bbox_;
    bool supports_safety_{};

    // Retain volumes since we save a pointer for debugging
    SPConstVolumes volumes_;

    // Host/device storage and reference
    ParamsDataStore<OrangeParamsData> data_;
};

//---------------------------------------------------------------------------//

extern template class ParamsDataStore<OrangeParamsData>;
extern template class ParamsDataInterface<OrangeParamsData>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Number of levels in the most deeply nested universe path.
 */
UnivLevelId::size_type OrangeParams::num_univ_levels() const
{
    return this->host_ref().scalars.num_univ_levels;
}

//---------------------------------------------------------------------------//
/*!
 * Get surface metadata.
 */
auto OrangeParams::surfaces() const -> SurfaceMap const&
{
    return impl_surf_labels_;
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
    return impl_vol_labels_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the canonical volume IDs corresponding to an implementation volume.
 */
VolumeId OrangeParams::volume_id(ImplVolumeId iv_id) const
{
    auto const& volume_id_map = this->host_ref().volume_ids;
    CELER_EXPECT(iv_id < volume_id_map.size());
    return volume_id_map[iv_id];
}

//---------------------------------------------------------------------------//
/*!
 * Get the canonical volume instance corresponding to an implementation volume.
 *
 * This may be null if the local volume corresponds to a "background" volume or
 * "outside".
 */
VolumeInstanceId OrangeParams::volume_instance_id(ImplVolumeId iv_id) const
{
    auto const& volume_inst_id_map = this->host_ref().volume_instance_ids;
    CELER_EXPECT(iv_id < volume_inst_id_map.size());
    return volume_inst_id_map[iv_id];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
