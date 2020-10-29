//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4VUserDetectorConstruction.hh>

namespace geant_exporter
{
//---------------------------------------------------------------------------//
/*!
 * Load the detector geometry from a GDML input file.
 */
class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    explicit DetectorConstruction(G4String gdmlInput);
    ~DetectorConstruction();

    G4VPhysicalVolume* Construct() override;
    const G4VPhysicalVolume* get_world_volume() const;

  private:
    std::unique_ptr<G4VPhysicalVolume> phys_vol_world_;
};

//---------------------------------------------------------------------------//
} // namespace geant_exporter
