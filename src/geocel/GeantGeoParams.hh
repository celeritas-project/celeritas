//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <string>

#include "corecel/Macros.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "BoundingBox.hh"
#include "GeoParamsInterface.hh"
#include "Types.hh"
#include "g4/GeantGeoData.hh"

#include "detail/GeantVolumeInstanceMapper.hh"

#if !CELERITAS_USE_GEANT4
#    include "inp/Model.hh"
#endif

class G4LogicalSurface;
class G4Material;
class G4VPhysicalVolume;
class G4VSensitiveDetector;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage and provide access to a Geant4 geometry model.
 *
 * This can be constructed directly by loading a GDML file, or in-memory using
 * an existing physical volume. The \c make_model_input function returns the
 * geometry hierarchy including surface definitions for optical physics. The \c
 * geant_to_id and \c id_to_geant functions provide mappings between Geant4
 * pointers and Celeritas IDs.
 *
 * The \c ImplVolumeId used by Celeritas is equal to the index of a \c
 * G4LogicalVolume in the \c G4LogicalVolumeStore. Due to potential resetting
 * of the geometry, the internal Geant4 ID for the volume may be
 * offset from this index. Currently the \c ImplVolumeId maps exactly to the \c
 * VolumeId value: some volumes may be unreachable by the world hierarchy.
 *
 * In general, the \c G4VPhysicalVolume is equivalent to the index in its
 * store. However, due to the way Geant4 represents "parameterised" and
 * "replicated" placements, a single G4PV may correspond to multiple spatial
 * placements. Celeritas disambiguates and maps each replicated instance to a
 * distinct \c VolumeInstanceId (see \c detail::GeantVolumeInstanceMapper).
 * When querying this ID from an in-memory physical volume, the returned value
 * uses the G4PV's \em current state (i.e., the copy number). Similarly,
 * calling \c id_to_geant on a volume instance ID for a replica volume will \em
 * change the thread-local state of the \c G4VPhysicalVolume.
 *
 * Each \c SurfaceId maps to a \c G4LogicalSurface instance, which is ether a
 * \c G4LogicalBorderSurface (an "interface" surface between two volume
 * instances) or a \c G4LogicalSkinSurface (a "boundary" surrounding a single
 * logical volume). To ensure reproducible surface IDs across runs, we put
 * boundaries before interfaces, and sort within each set by volume IDs (not by
 * Geant4 object pointers, which is what the Geant4 implementation stores in a
 * table).  Surface labels are accessed via the SurfaceParams object, which can
 * be created by the model input returned by this class.
 *
 * When running standalone for testing, a "secret" environment flag
 * \c G4_GEO_OPTIMIZE can disable Geant4 "voxelization" (acceleration structure
 * setup) which can be slow for large problems and is unnecessary if \em not
 * tracking on the Geant4 geometry.
 *
 * \todo Much of the conversion should be factored out into a separate Model
 * class which provides adapters to materials, detectors, and geometry
 * structure.
 */
class GeantGeoParams final : public GeoParamsInterface,
                             public ParamsDataInterface<GeantGeoParamsData>
{
  public:
    // Create from a running Geant4 application
    static std::shared_ptr<GeantGeoParams> from_tracking_manager();

    // Create from a GDML file
    static std::shared_ptr<GeantGeoParams>
    from_gdml(std::string const& filename);

    // Return the input geometry for a consistent interface
    inline static std::shared_ptr<GeantGeoParams>
    from_geant(std::shared_ptr<GeantGeoParams const> const& geo);

    // Create a VecGeom model from an already-loaded Geant4 geometry
    // TODO: also take model input? see #1815
    GeantGeoParams(G4VPhysicalVolume const* world, Ownership owns);

    CELER_DEFAULT_MOVE_DELETE_COPY(GeantGeoParams);

    // Clean up on destruction
    ~GeantGeoParams() final;

    //// GEOMETRY INTERFACE ////

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return true; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    // Create model parameters corresponding to our internal representation
    inp::Model make_model_input() const final;

    //// VOLUMES ////

    // Get (logical) volume metadata
    inline ImplVolumeMap const& impl_volumes() const final;

    // Get the Geant4 physical volume corresponding to a volume instance ID
    inline G4VPhysicalVolume const* id_to_geant(VolumeInstanceId vol_id) const;

    // Get the Geant4 logical volume corresponding to a volume ID
    G4LogicalVolume const* id_to_geant(VolumeId vol_id) const;

    // Get the canonical volume IDs corresponding to an implementation volume
    inline VolumeId volume_id(ImplVolumeId) const final;

    // Get the volume instance containing the global point
    VolumeInstanceId find_volume_instance_at(Real3 const&) const final;

    //// SURFACES ////

    //! Get the number of surfaces (TODO: maybe live in surface params?)
    SurfaceId::size_type num_surfaces() const { return surfaces_.size(); }

    // Get the Geant4 logical surface corresponding to a surface ID
    inline G4LogicalSurface const* id_to_geant(SurfaceId surf_id) const;

    // DEPRECATED

    //! Offset of logical volume ID after reloading geometry
    ImplVolumeId::size_type lv_offset() const { return data_.lv_offset; }

    //! Offset of material index after reloading geometry
    GeoMatId::size_type mat_offset() const { return data_.mat_offset; }

    //// G4 ACCESSORS ////

    // Get the geometry material ID for a logical volume
    GeoMatId geant_to_id(G4Material const& mat) const;

    // Get the canonical volume ID corresponding to a Geant4 logical volume
    VolumeId geant_to_id(G4LogicalVolume const& volume) const;

    // Get the volume instance ID corresponding to a Geant4 physical volume
    inline VolumeInstanceId geant_to_id(G4VPhysicalVolume const& volume) const;

    //!@{
    //! Access the world volume
    G4VPhysicalVolume const* world() const { return data_.world; }
    G4VPhysicalVolume* world() { return data_.world; }
    //!@}

    // Get the world extents in Geant4 units
    BoundingBox<double> get_clhep_bbox() const;

    //// DATA ACCESS ////

    //! Access geometry data on host
    HostRef const& host_ref() const final { return data_; }

    //! No GPU support code
    DeviceRef const& device_ref() const final
    {
        CELER_NOT_IMPLEMENTED("Geant4 on GPU");
    }

  private:
    using MapStrDetector
        = std::map<std::string, std::shared_ptr<G4VSensitiveDetector>>;

    //// DATA ////

    Ownership ownership_{Ownership::reference};
    bool closed_geometry_{false};
    MapStrDetector built_detectors_;

    // Host metadata/access
    ImplVolumeMap impl_volumes_;
    detail::GeantVolumeInstanceMapper vi_mapper_;
    std::vector<G4LogicalSurface const*> surfaces_;
    BBox bbox_;

    // Storage
    HostRef data_;

    //// HELPER FUNCTIONS ////

    // Construct labels and other host-only metadata
    void build_metadata();
};

//---------------------------------------------------------------------------//
// Set non-owning reference to global tracking geometry instance
void global_geant_geo(std::shared_ptr<GeantGeoParams const> const&);

// Global tracking geometry instance: may be nullptr
std::weak_ptr<GeantGeoParams const> const& global_geant_geo();

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Return the input geometry for a consistent interface.
 */
std::shared_ptr<GeantGeoParams>
GeantGeoParams::from_geant(std::shared_ptr<GeantGeoParams const> const& geo)
{
    CELER_EXPECT(geo);
    return std::const_pointer_cast<GeantGeoParams>(geo);
}

//---------------------------------------------------------------------------//
/*!
 * Get volume metadata.
 *
 * Volumes correspond directly to Geant4 logical volumes.
 */
auto GeantGeoParams::impl_volumes() const -> ImplVolumeMap const&
{
    return impl_volumes_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 physical volume corresponding to a volume instance ID.
 */
CELER_FORCEINLINE G4VPhysicalVolume const*
GeantGeoParams::id_to_geant(VolumeInstanceId id) const
{
    return &vi_mapper_.id_to_geant(id);
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 logical surface corresponding to a surface ID.
 */
G4LogicalSurface const* GeantGeoParams::id_to_geant(SurfaceId id) const
{
    CELER_EXPECT(!id || id < surfaces_.size());
    if (CELER_UNLIKELY(!id))
    {
        return nullptr;
    }

    return surfaces_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance ID corresponding to a Geant4 physical volume.
 *
 * \warning For Geant4 parameterised/replicated volumes, external state (e.g.
 * the local navigation) \em must be used in concert with this method: i.e.,
 * navigation on the current thread needs to have just "visited" the instance.
 */
CELER_FORCEINLINE VolumeInstanceId
GeantGeoParams::geant_to_id(G4VPhysicalVolume const& volume) const
{
    return vi_mapper_.geant_to_id(volume);
}

//---------------------------------------------------------------------------//
/*!
 * Get the canonical volume IDs corresponding to an implementation volume.
 *
 * \note See make_inp_volumes : for now, volume IDs and impl IDs are identical
 */
VolumeId GeantGeoParams::volume_id(ImplVolumeId iv_id) const
{
    if (CELER_UNLIKELY(!iv_id))
    {
        return {};
    }

    return id_cast<VolumeId>(iv_id.get());
}

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4 && !defined(__DOXYGEN__)
inline void global_geant_geo(std::shared_ptr<GeantGeoParams const> const&)
{
    CELER_ASSERT_UNREACHABLE();
}
inline std::weak_ptr<GeantGeoParams const> const& global_geant_geo()
{
    static std::weak_ptr<GeantGeoParams const> const temp_;
    return temp_;
}
//-----------------------------------//
inline std::shared_ptr<GeantGeoParams> GeantGeoParams::from_tracking_manager()
{
    return std::make_shared<GeantGeoParams>(nullptr, Ownership::reference);
}

inline std::shared_ptr<GeantGeoParams>
GeantGeoParams::from_gdml(std::string const&)
{
    return std::make_shared<GeantGeoParams>(nullptr, Ownership::reference);
}

inline GeantGeoParams::GeantGeoParams(G4VPhysicalVolume const*, Ownership)
{
    CELER_NOT_CONFIGURED("Geant4");
}

//-----------------------------------//
inline GeantGeoParams::~GeantGeoParams() = default;

//-----------------------------------//
inline inp::Model GeantGeoParams::make_model_input() const
{
    CELER_ASSERT_UNREACHABLE();
}
inline G4LogicalVolume const* GeantGeoParams::id_to_geant(VolumeId) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline GeoMatId GeantGeoParams::geant_to_id(G4Material const&) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline VolumeId GeantGeoParams::geant_to_id(G4LogicalVolume const&) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline VolumeInstanceId
GeantGeoParams::find_volume_instance_at(Real3 const&) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline BoundingBox<double> GeantGeoParams::get_clhep_bbox() const
{
    CELER_ASSERT_UNREACHABLE();
}

#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
