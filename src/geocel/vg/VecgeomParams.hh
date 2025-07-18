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
    //!@}

  public:
    // Whether surface tracking is being used
    static bool use_surface_tracking();

    // Whether VecGeom GDML is being used to load the geometry
    static bool use_vgdml();

    //!@{
    //! \name Static constructor helpers
    //! \todo: move these to a "model" abstraction that loads/emits geometry,
    //! materials, volumes?

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
    VecgeomParams(vecgeom::GeoManager const&, Ownership, VecLv const&, VecPv&&);

    // Clean up VecGeom on destruction
    ~VecgeomParams() final;

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return true; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    //! Maximum nested geometry depth
    //! \todo move to VolumeParams
    LevelId::size_type max_depth() const final { return host_ref_.max_depth; }

    // Create model parameters corresponding to our internal representation
    inp::Model make_model_input() const final;

    //// VOLUMES ////

    // Get volume metadata
    inline VolumeMap const& volumes() const final;

    // Get (physical) volume instance metadata
    inline VolInstanceMap const& volume_instances() const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    VolumeId find_volume(G4LogicalVolume const* volume) const final;

    // Get the Geant4 physical volume corresponding to a volume instance ID
    GeantPhysicalInstance id_to_geant(VolumeInstanceId vol_id) const final;

    // DEPRECATED
    using GeoParamsInterface::find_volume;

    //// DATA ACCESS ////

    //! Access geometry data on host
    HostRef const& host_ref() const final { return host_ref_; }

    //! Access geometry data on device
    DeviceRef const& device_ref() const final { return device_ref_; }

  private:
    //// DATA ////

    // Flag for resetting VecGeom on destruction
    Ownership ownership_{Ownership::reference};

    // Host metadata/access
    LabelIdMultiMap<VolumeId> volumes_;
    VolInstanceMap vol_instances_;
    std::unordered_map<G4LogicalVolume const*, VolumeId> g4log_volid_map_;
    std::vector<G4VPhysicalVolume const*> g4_pv_map_;

    BBox bbox_;

    // Host/device storage and reference
    HostRef host_ref_;
    DeviceRef device_ref_;

    //// HELPER FUNCTIONS ////

    // Construct VecGeom tracking data and copy to GPU
    void build_surface_tracking();
    void build_volume_tracking();
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
