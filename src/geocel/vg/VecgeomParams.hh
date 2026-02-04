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

#include "corecel/Config.hh"

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/Types.hh"
#include "geocel/inp/Model.hh"

#include "VecgeomData.hh"

class G4VPhysicalVolume;

// clang-format off
#if CELERITAS_VECGEOM_VERSION < 0x020000
#  if defined(__CUDACC__) || defined(__NVCC__)
#    define CELERITAS_VECGEOM_GM_CXX_NS_BEGIN namespace cxx {
#  else
#    define CELERITAS_VECGEOM_GM_CXX_NS_BEGIN inline namespace cxx {
#  endif
#  define CELERITAS_VECGEOM_GM_CXX_NS_END } // end namespace cxx
#else
#  define CELERITAS_VECGEOM_GM_CXX_NS_BEGIN
#  define CELERITAS_VECGEOM_GM_CXX_NS_END
#endif
// clang-format on

namespace vecgeom
{
CELERITAS_VECGEOM_GM_CXX_NS_BEGIN
class GeoManager;
CELERITAS_VECGEOM_GM_CXX_NS_END
}  // namespace vecgeom

namespace celeritas
{
class GeantGeoParams;
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
    //!@{
    //! \name Type aliases
    using VecLv = std::vector<G4LogicalVolume const*>;
    using VecPv = std::vector<G4VPhysicalVolume const*>;
    using ImplVolInstanceId = VgVolumeInstanceId;
    using ImplVolInstanceMap = LabelIdMultiMap<ImplVolInstanceId>;
    //!@}

  public:
    // Whether surface tracking is being used
    static bool use_surface_tracking();

    // Whether VecGeom GDML is being used to load the geometry
    static bool use_vgdml();

    //!@{
    //! \name Static constructor helpers
    //! \todo Move these to a "model" abstraction

    // Build by loading a GDML file (best available method)
    static std::shared_ptr<VecgeomParams>
    from_gdml(std::string const& filename);

    // Build by loading a GDML file through Geant4
    static std::shared_ptr<VecgeomParams>
    from_gdml_g4(std::string const& filename);

    // Build by loading a GDML file through VecGeom VGDML
    static std::shared_ptr<VecgeomParams>
    from_gdml_vg(std::string const& filename);

    // Build from a Geant4 geometry
    static std::shared_ptr<VecgeomParams>
    from_geant(std::shared_ptr<GeantGeoParams const> const& geo);

    //!@}

    // Build from existing geometry, with ownership and mappings
    VecgeomParams(vecgeom::GeoManager const&,
                  Ownership,
                  VecLv const&,
                  VecPv const&);

    // Clean up VecGeom on destruction
    ~VecgeomParams() final;

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return true; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    // Maximum nested geometry depth
    inline VolumeLevelId::size_type num_volume_levels() const;

    // Create model parameters corresponding to our internal representation
    inp::Model make_model_input() const final;

    //// VOLUMES ////

    // Get volume metadata for VG logical volumes
    inline ImplVolumeMap const& impl_volumes() const final;

    // Get volume metadata for VG placed volumes
    inline ImplVolInstanceMap const& impl_volume_instances() const;

    // Get the canonical volume IDs corresponding to an implementation volume
    inline VolumeId volume_id(ImplVolumeId) const final;

    //// DATA ACCESS ////

    //! Access geometry data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access geometry data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    //// DATA ////

    // Flag for resetting VecGeom on destruction
    Ownership host_ownership_{Ownership::reference};
    Ownership device_ownership_{Ownership::reference};

    // Geant4 model used to construct
    std::shared_ptr<GeantGeoParams const> geant_geo_;

    // Host metadata/access
    LabelIdMultiMap<ImplVolumeId> impl_volumes_;
    ImplVolInstanceMap impl_vol_instances_;

    BBox bbox_;

    // Host/device storage and reference
    ParamsDataStore<VecgeomParamsData> data_;

    //// HELPER FUNCTIONS ////

    // Construct VecGeom tracking data and copy to GPU
    void build_surface_tracking();
    void build_volume_tracking();
};

//---------------------------------------------------------------------------//

extern template class ParamsDataStore<VecgeomParamsData>;
extern template class ParamsDataInterface<VecgeomParamsData>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Maximum nested geometry depth.
 *
 * \todo Only use in VolumeParams
 */
VolumeLevelId::size_type VecgeomParams::num_volume_levels() const
{
    return this->host_ref().scalars.num_volume_levels;
}
//
//---------------------------------------------------------------------------//
/*!
 * Get volume metadata.
 */
auto VecgeomParams::impl_volumes() const -> ImplVolumeMap const&
{
    return impl_volumes_;
}

//---------------------------------------------------------------------------//
/*!
 * Get volume instance metadata.
 *
 * Volume instances correspond directly to Geant4 physical volumes.
 */
auto VecgeomParams::impl_volume_instances() const -> ImplVolInstanceMap const&
{
    return impl_vol_instances_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the canonical volume IDs corresponding to an implementation volume.
 *
 * \note See make_inp_volumes : for now, volume IDs and impl IDs are identical
 */
inline VolumeId VecgeomParams::volume_id(ImplVolumeId iv_id) const
{
    auto const& vol_ids = this->host_ref().volumes;
    CELER_EXPECT(!vol_ids.empty());

    return vol_ids[iv_id];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
