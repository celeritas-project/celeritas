//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/GeantLoadTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Test.hh"

class G4VPhysicalVolume;

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Load a Geant4 geometry and clean up as needed.
 */
class GeantLoadTestBase : public ::celeritas::test::Test
{
  public:
    // Build via Geant4 GDML reader
    G4VPhysicalVolume const* load_gdml(std::string const& filename);

    // Load a test input
    G4VPhysicalVolume const* load_test_gdml(std::string_view basename);

    // Reset the geometry
    static void TearDownTestSuite();

  private:
    static std::string loaded_filename_;
    static G4VPhysicalVolume* world_volume_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
