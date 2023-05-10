//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <string>
#include <G4VUserDetectorConstruction.hh>

class G4LogicalVolume;

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct a detector from a GDML filename set in GlobalSetup.
 */
class DetectorConstruction final : public G4VUserDetectorConstruction
{
  public:
    // Set up global celeritas SD options during construction
    DetectorConstruction();

    G4VPhysicalVolume* Construct() final;
    void ConstructSDandField() final;

  private:
    std::unique_ptr<G4VPhysicalVolume> world_;
    std::multimap<std::string, G4LogicalVolume*> detectors_;
};

//---------------------------------------------------------------------------//
}  // namespace demo_geant
