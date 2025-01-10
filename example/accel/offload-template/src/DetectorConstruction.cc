//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/accel/offload-template/src/DetectorConstruction.cc
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

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
DetectorConstruction::DetectorConstruction() : G4VUserDetectorConstruction() {}

//---------------------------------------------------------------------------//
/*!
 * Example geometry is a Pb world box with 10 m side.
 */
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    // World material
    auto const nist = G4NistManager::Instance();
    auto const world_material = nist->FindOrBuildMaterial("G4_Pb");

    // World solid
    double const world_size = 10 * m;
    auto world_box = new G4Box("world_box", world_size, world_size, world_size);

    // World logical volume
    auto world_lv = new G4LogicalVolume(world_box, world_material, "world_lv");

    // World physical volume
    auto world_pv = new G4PVPlacement(
        nullptr, G4ThreeVector(), world_lv, "world_pv", nullptr, false, 0);

    return world_pv;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize sensitive detectors. This is the only Celeritas interface.
 */
void DetectorConstruction::ConstructSDandField()
{
    auto world_sd = new SensitiveDetector("world_sd");
    G4SDManager::GetSDMpointer()->AddNewDetector(world_sd);
    G4VUserDetectorConstruction::SetSensitiveDetector("world_lv", world_sd);
}
