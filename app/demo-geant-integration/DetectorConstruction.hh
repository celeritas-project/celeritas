//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/DetectorConstruction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4VUserDetectorConstruction.hh>
#include "celeritas/ext/LoadGdml.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Construct a detector from a GDML filename.
 */
class DetectorConstruction final : public G4VUserDetectorConstruction
{
  public:
    explicit DetectorConstruction(const std::string& filename)
    {
        world_ = celeritas::load_gdml(filename);
    }

    G4VPhysicalVolume* Construct() final;
    void               ConstructSDandField() final;

  private:
    celeritas::UPG4PhysicalVolume world_;
};

//---------------------------------------------------------------------------//
} // namespace demo_geant
