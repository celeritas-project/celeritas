//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/LazyGeantGeoManager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "geocel/GeoParamsInterface.hh"

namespace celeritas
{
class GeantGeoParams;
class VolumeParams;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Base class for managing a persistent geometry built from Geant4.
 *
 * Daughter classes must implement \c build_from_geant , and \c gdml_basename
 * must return a filename prefix to a GDML file in `geocel/test`.
 */
class LazyGeantGeoManager
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstGeoI = std::shared_ptr<GeoParamsInterface const>;
    using SPConstGeantGeo = std::shared_ptr<GeantGeoParams const>;
    using SPConstVolumes = std::shared_ptr<VolumeParams const>;
    //!@}

  public:
    //! Get an identifying key for the geometry (basename, description, etc)
    virtual std::string_view gdml_basename() const = 0;

    // Implementation builds Geant4 geometry on request using GDML by default
    [[nodiscard]] virtual SPConstGeantGeo
    build_geant_geo(std::string const& filename) const;

    // Implementation builds from Geant4 on request
    [[nodiscard]] virtual SPConstGeoI
    build_geo_from_geant(SPConstGeantGeo const&) const
        = 0;

    // Backup method when Geant4 is disabled
    [[nodiscard]] virtual SPConstGeoI
    build_geo_from_gdml(std::string const& filename) const;

    //// ACCESSORS ////

    // Access the basename of the geometry that's currently cached
    std::string const& cached_gdml_basename() const;

    // Construct or access a geometry
    SPConstGeoI lazy_geo() const;

    // Access Geant4 geometry if already built (null if invalid)
    SPConstGeantGeo geant_geo() const;

    // Access volumes from built geometry or geant4 model
    SPConstVolumes volumes() const;

    // Reset geometry (not G4) manually; needed by AllGeoTypedTestBase
    static void clear_lazy_geo();
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
