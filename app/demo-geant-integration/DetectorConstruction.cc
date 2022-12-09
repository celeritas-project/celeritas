//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include <G4GDMLParser.hh>
#include <G4LogicalVolume.hh>
#include <G4SDManager.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/io/Logger.hh"

#include "SensitiveDetector.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Load geometry and sensitive detector volumes.
 */
DetectorConstruction::DetectorConstruction(const std::string& filename)
{
    // Create parser; do *not* strip `0x` extensions since those are needed to
    // deduplicate complex geometries (e.g. CMS) and are handled by the Label
    // and LabelIdMultiMap. Note that material and element names (at least as
    // of Geant4@11.0) are *always* stripped: only volumes and solids keep
    // their extension.
    G4GDMLParser gdml_parser;
    gdml_parser.SetStripFlag(false);

    constexpr bool validate_gdml_schema = false;
    gdml_parser.Read(filename, validate_gdml_schema);

    // Claim ownership of world volume
    world_.reset(gdml_parser.GetWorldVolume());

    // Find sensitive detectors
    for (const auto& lv_vecaux : *gdml_parser.GetAuxMap())
    {
        for (const G4GDMLAuxStructType& aux : lv_vecaux.second)
        {
            if (aux.type == "SensDet")
            {
                detectors_.emplace_back(lv_vecaux.first, aux.value);
            }
        }
    }
}

//---------------------------------------------------------------------------//
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    CELER_LOG_LOCAL(debug) << "DetectorConstruction::Construct";
    return world_.release();
}

//---------------------------------------------------------------------------//
void DetectorConstruction::ConstructSDandField()
{
    CELER_LOG_LOCAL(debug) << "DetectorConstruction::ConstructSDandField";

    G4SDManager* sd_manager = G4SDManager::GetSDMpointer();

    for (auto& lv_name : detectors_)
    {
        auto detector = std::make_unique<SensitiveDetector>(lv_name.second);
        lv_name.first->SetSensitiveDetector(detector.get());
        sd_manager->AddNewDetector(detector.release());
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
