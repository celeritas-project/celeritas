//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
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
    DetectorConstruction() = default;

    G4VPhysicalVolume* Construct() final;
    void               ConstructSDandField() final;

  private:
    std::unique_ptr<G4VPhysicalVolume>                    world_;
    std::vector<std::pair<G4LogicalVolume*, std::string>> detectors_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
