//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Macros.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "BoundingBox.hh"
#include "GeoParamsInterface.hh"
#include "Types.hh"
#include "g4/GeantGeoData.hh"

#if !CELERITAS_USE_GEANT4
#    include "inp/Model.hh"
#endif

class G4VPhysicalVolume;
class G4LogicalSurface;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared Geant4 geometry model wrapper.
 *
 * This can be constructed directly by loading a GDML file, or in-memory using
 * an existing physical volume. The \c make_model_input function returns the
 * geometry hierarchy including surface definitions for optical physics.
 *
 * The \c VolumeId used by Celeritas is equal to the index of a \c
 * G4LogicalVolume in the \c G4LogicalVolumeStore. Due to potential resetting
 * of the geometry, the "volume instance ID" for the logical volume may be
 * offset from this index.
 *
 * Analogously, the \c G4VPhysicalVolume is equivalent to the index in its
 * store. Due to the way Geant4 represents "parameterised" and "replicated"
 * placements, a single PV may correspond to multiple spatial placements and is
 * disambiguated with \c ReplicaId , which corresponds to the PV's "copy
 * number".
 *
 * Each \c SurfaceId maps to a \c G4LogicalSurface instance, which is ether a
 * \c G4LogicalBorderSurface (an "interface" surface between two volume
 * instances) or a \c G4LogicalSkinSurface (a "boundary" surrounding a single
 * logical volume). To ensure reproducible surface IDs across runs, we put
 * boundaries before interfaces, and sort within each set by volume IDs (not by
 * Geant4 object pointers, which is what the Geant4 implementation stores in a
 * table).  Surface labels are accessed via the SurfaceParams object, which can
 * be created by the model input returned by this class.
 */
class GeantGeoParams final : public GeoParamsInterface,
                             public ParamsDataInterface<GeantGeoParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using ReplicaId = GeantPhysicalInstance::ReplicaId;
    //!@}

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

    // Maximum nested scene/volume depth
    LevelId::size_type max_depth() const { return max_depth_; }

    // Create model parameters corresponding to our internal representation
    inp::Model make_model_input() const final;

    //// VOLUMES ////

    // Get (logical) volume metadata
    inline VolumeMap const& volumes() const final;

    // Get (physical) volume instance metadata
    inline VolInstanceMap const& volume_instances() const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    VolumeId find_volume(G4LogicalVolume const* volume) const final;

    // Get the Geant4 physical volume corresponding to a volume instance ID
    GeantPhysicalInstance id_to_geant(VolumeInstanceId vol_id) const final;

    // Get the Geant4 logical volume corresponding to a volume ID
    G4LogicalVolume const* id_to_geant(VolumeId vol_id) const;

    //// SURFACES ////

    //! Get the number of surfaces (TODO: maybe live in surface params?)
    SurfaceId::size_type num_surfaces() const { return surfaces_.size(); }

    // Get the Geant4 logical surface corresponding to a surface ID
    inline G4LogicalSurface const* id_to_geant(SurfaceId surf_id) const;

    // DEPRECATED
    using GeoParamsInterface::find_volume;

    //! Offset of logical volume ID after reloading geometry
    VolumeId::size_type lv_offset() const { return data_.lv_offset; }

    //! Offset of physical volume ID after reloading geometry
    VolumeInstanceId::size_type pv_offset() const { return data_.pv_offset; }

    //! Offset of material index after reloading geometry
    GeoMatId::size_type mat_offset() const { return data_.mat_offset; }

    //// G4 ACCESSORS ////

    //! Get the volume ID corresponding to a Geant4 logical volume
    VolumeId geant_to_id(G4LogicalVolume const& volume) const
    {
        return this->find_volume(&volume);
    }

    // Get the volume ID corresponding to a Geant4 physical volume
    VolumeInstanceId geant_to_id(G4VPhysicalVolume const& volume) const;

    // Get the replica ID corresponding to a Geant4 physical volume
    ReplicaId replica_id(G4VPhysicalVolume const& volume) const;

    //!@{
    //! Access the world volume
    G4VPhysicalVolume const* world() const { return data_.world; }
    G4VPhysicalVolume* world() { return data_.world; }
    //!@}

    // Get the world extents in Geant4 units
    BoundingBox<double> get_clhep_bbox() const;

    // Initialize thread-local mutable copy numbers for "replica" volumes
    void reset_replica_data() const;

    //// DATA ACCESS ////

    //! Access geometry data on host
    HostRef const& host_ref() const final { return data_; }

    //! No GPU support code
    DeviceRef const& device_ref() const final
    {
        CELER_NOT_IMPLEMENTED("Geant4 on GPU");
    }

  private:
    //// DATA ////

    Ownership ownership_{Ownership::reference};
    bool closed_geometry_{false};

    // Host metadata/access
    VolumeMap volumes_;
    VolInstanceMap vol_instances_;
    std::vector<G4LogicalSurface const*> surfaces_;
    BBox bbox_;
    LevelId::size_type max_depth_{0};

    // Storage
    HostRef data_;

    //// HELPER FUNCTIONS ////

    // Construct labels and other host-only metadata
    void build_metadata();
};

//---------------------------------------------------------------------------//
// Set non-owning reference to global tracking geometry instance
void geant_geo(std::shared_ptr<GeantGeoParams const> const&);

// Global tracking geometry instance: may be nullptr
std::weak_ptr<GeantGeoParams const> const& geant_geo();

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
auto GeantGeoParams::volumes() const -> VolumeMap const&
{
    return volumes_;
}

//---------------------------------------------------------------------------//
/*!
 * Get volume instance metadata.
 *
 * Volume instances correspond directly to Geant4 physical volumes.
 */
auto GeantGeoParams::volume_instances() const -> VolInstanceMap const&
{
    return vol_instances_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 logical surface corresponding to a surface ID.
 */
G4LogicalSurface const* GeantGeoParams::id_to_geant(SurfaceId id) const
{
    CELER_EXPECT(!id || id < surfaces_.size());
    if (!id)
    {
        return {};
    }

    return surfaces_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_GEANT4 && !defined(__DOXYGEN__)
inline void geant_geo(std::shared_ptr<GeantGeoParams const> const&)
{
    CELER_ASSERT_UNREACHABLE();
}
inline std::weak_ptr<GeantGeoParams const> const& geant_geo()
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
inline VolumeId GeantGeoParams::find_volume(G4LogicalVolume const*) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline GeantPhysicalInstance GeantGeoParams::id_to_geant(VolumeInstanceId) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline G4LogicalVolume const* GeantGeoParams::id_to_geant(VolumeId) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline VolumeInstanceId
GeantGeoParams::geant_to_id(G4VPhysicalVolume const&) const
{
    CELER_ASSERT_UNREACHABLE();
}
inline GeantGeoParams::ReplicaId
GeantGeoParams::replica_id(G4VPhysicalVolume const&) const
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
