//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/GeantLoadTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "geocel/LazyGeoManager.hh"

#include "Test.hh"

class G4VPhysicalVolume;

namespace celeritas
{
class GeantGeoParams;
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Load a Geant4 geometry and clean up as needed.
 */
class GeantLoadTestBase : public ::celeritas::test::Test,
                          public ::celeritas::test::LazyGeoManager
{
  public:
    using SPConstGeo = std::shared_ptr<GeantGeoParams const>;

    // Build via Geant4 GDML reader
    void load_gdml(std::string const& filename);

    // Load a test input
    void load_test_gdml(std::string_view basename);

    // Access the geo params after loading
    GeantGeoParams const& geo() const;

    // Access the world volume after loading
    G4VPhysicalVolume const& world() const;

  protected:
    // Construct a fresh geometry from a filename
    SPConstGeoI build_fresh_geometry(std::string_view key) final;

    SPConstGeo geo_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
