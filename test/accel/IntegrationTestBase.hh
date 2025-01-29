//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Quantities.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "Test.hh"

class G4VUserDetectorConstruction;
class G4VUserPrimaryGeneratorAction;
class G4RunManager;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for integration tests.
 */
class IntegrationTestBase : public ::celeritas::test::Test
{
  public:
    //!@{
    //! \name Type aliases
    using UPDetector = std::unique_ptr<G4VUserDetectorConstruction>;
    using UPPrimary = std::unique_ptr<G4VUserPrimaryGeneratorAction>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Create or get the Geant4 run manager
    static G4RunManager& run_manager();

    // Create geometry helper
    static UPDetector make_detector_construction();

    // Create primary generator, isotropic at origin
    static UPPrimary make_primaries(PDGNumber pdg, Energy energy);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
