//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserDetectorConstruction.hh>

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Construct detector geometry.
 */
class DetectorConstruction final : public G4VUserDetectorConstruction
{
  public:
    // Construct empty
    DetectorConstruction();

    // Define programmatic geometry
    G4VPhysicalVolume* Construct() final;

    // Sensitive detectors are the only Celeritas interface with Geant4
    void ConstructSDandField() final;
};

}  // namespace example
}  // namespace celeritas
