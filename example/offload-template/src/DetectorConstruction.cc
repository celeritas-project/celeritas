//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file offload-template/src/DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include <G4Box.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4SDManager.hh>
#include <G4SystemOfUnits.hh>
#include <G4VPhysicalVolume.hh>

#include "SensitiveDetector.hh"

namespace celeritas
{
namespace example
{
//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
DetectorConstruction::DetectorConstruction() : G4VUserDetectorConstruction() {}

//---------------------------------------------------------------------------//
/*!
 * Generate example geometry.
 */
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    // Construct single material world volume box
    auto* nist = G4NistManager::Instance();
    auto* world_material = nist->FindOrBuildMaterial("G4_Pb");
    double const world_size = 1 * m;
    auto* world_box
        = new G4Box("world_box", world_size, world_size, world_size);
    auto* world_lv = new G4LogicalVolume(world_box, world_material, "world_lv");
    auto* world_pv = new G4PVPlacement(
        nullptr, G4ThreeVector(), world_lv, "world_pv", nullptr, false, 0);

    return world_pv;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize sensitive detectors.
 *
 * Every volume that needs to collect data from Celeritas *must* be defined as
 * a sensitive detector.
 *
 * \sa \c SensitiveDetector::ProcessHits from this example.
 */
void DetectorConstruction::ConstructSDandField()
{
    auto* world_sd = new SensitiveDetector("world_sd");
    G4SDManager::GetSDMpointer()->AddNewDetector(world_sd);
    G4VUserDetectorConstruction::SetSensitiveDetector("world_lv", world_sd);
}

}  // namespace example
}  // namespace celeritas
