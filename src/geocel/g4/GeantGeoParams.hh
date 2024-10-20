//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Types.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "geocel/BoundingBox.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/Types.hh"

#include "GeantGeoData.hh"

class G4VPhysicalVolume;

namespace celeritas
{
//---------------------------------------------------------------------------//
class ScopedGeantLogger;
class ScopedGeantExceptionHandler;

//---------------------------------------------------------------------------//
/*!
 * Shared Geant4 geometry model wrapper.
 *
 * This can be constructed directly by loading a GDML file, or in-memory using
 * an existing physical volume. One "gotcha" is that due to persistent static
 * variables in Geant4, the volume IDs will be offset if a geometry has been
 * loaded and closed previously.
 */
class GeantGeoParams final : public GeoParamsInterface,
                             public ParamsDataInterface<GeantGeoParamsData>
{
  public:
    // Construct from a GDML filename
    explicit GeantGeoParams(std::string const& gdml_filename);

    // Create a VecGeom model from a pre-existing Geant4 geometry
    explicit GeantGeoParams(G4VPhysicalVolume const* world);

    // Clean up on destruction
    ~GeantGeoParams();

    //! Access the world volume
    G4VPhysicalVolume const* world() const { return host_ref_.world; }

    //! Whether safety distance calculations are accurate and precise
    bool supports_safety() const final { return true; }

    //! Outer bounding box of geometry
    BBox const& bbox() const final { return bbox_; }

    //// VOLUMES ////

    // Get volume metadata
    inline VolumeMap const& volumes() const final;

    // Get the volume ID corresponding to a Geant4 logical volume
    VolumeId find_volume(G4LogicalVolume const* volume) const final;

    // Get the Geant4 logical volume corresponding to a volume ID
    G4LogicalVolume const* id_to_lv(VolumeId vol_id) const;

    // DEPRECATED
    using GeoParamsInterface::find_volume;

    //// DATA ACCESS ////

    //! Access geometry data on host
    HostRef const& host_ref() const final { return host_ref_; }

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
    BBox bbox_;

    // Host/device storage and reference
    HostRef host_ref_;

    //// HELPER FUNCTIONS ////

    // Complete geometry construction
    void build_tracking();

    // Construct labels and other host-only metadata
    void build_metadata();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get volume metadata.
 */
auto GeantGeoParams::volumes() const -> VolumeMap const&
{
    return volumes_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
