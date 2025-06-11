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
#include "ScopedGeantExceptionHandler.hh"
#include "ScopedGeantLogger.hh"
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

    // Construct from a GDML filename
    explicit GeantGeoParams(std::string const& gdml_filename);

    // Create a VecGeom model from an already-loaded Geant4 geometry
    explicit GeantGeoParams(G4VPhysicalVolume const* world);

    CELER_DEFAULT_MOVE_DELETE_COPY(GeantGeoParams);

    // Clean up on destruction
    ~GeantGeoParams() final;

    //// GEOMETRY INTERFACE ////

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return true; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    // Maximum nested scene/volume depth
    LevelId::size_type max_depth() const final { return max_depth_; }

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

    bool loaded_gdml_{false};
    bool closed_geometry_{false};
    std::unique_ptr<ScopedGeantLogger> scoped_logger_;
    std::unique_ptr<ScopedGeantExceptionHandler> scoped_exceptions_;

    // Host metadata/access
    VolumeMap volumes_;
    VolInstanceMap vol_instances_;
    BBox bbox_;
    LevelId::size_type max_depth_{0};

    // Storage
    HostRef data_;

    //// HELPER FUNCTIONS ////

    // Complete geometry construction
    void build_tracking();

    // Construct labels and other host-only metadata
    void build_metadata();
};

//---------------------------------------------------------------------------//
// Set non-owning reference to global tracking geometry instance
void geant_geo(GeantGeoParams const&);

// Global tracking geometry instance: may be nullptr
GeantGeoParams const* geant_geo();

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
#if !CELERITAS_USE_GEANT4 && !defined(__DOXYGEN__)
inline void geant_geo(GeantGeoParams const&)
{
    CELER_ASSERT_UNREACHABLE();
}
inline GeantGeoParams const* geant_geo()
{
    CELER_ASSERT_UNREACHABLE();
}
//-----------------------------------//

inline GeantGeoParams::GeantGeoParams(G4VPhysicalVolume const*)
{
    CELER_NOT_CONFIGURED("Geant4");
}
inline GeantGeoParams::GeantGeoParams(std::string const&)
{
    CELER_NOT_CONFIGURED("Geant4");
}
inline std::shared_ptr<GeantGeoParams>
GeantGeoParams::GeantGeoParams::from_tracking_manager()
{
    CELER_NOT_CONFIGURED("Geant4");
}

//-----------------------------------//
inline GeantGeoParams::~GeantGeoParams() {}

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
