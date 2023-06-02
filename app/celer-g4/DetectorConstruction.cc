//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include <map>
#include <G4Exception.hh>
#include <G4GDMLAuxStructType.hh>
#include <G4GDMLParser.hh>
#include <G4LogicalVolume.hh>
#include <G4SDManager.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/io/Logger.hh"
#include "accel/SetupOptions.hh"

#include "GlobalSetup.hh"
#include "SensitiveDetector.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas SD options during construction.
 *
 * This should be done only during the main/serial thread.
 */
DetectorConstruction::DetectorConstruction()
{
    auto& sd = celeritas::app::GlobalSetup::Instance()->GetSDSetupOptions();

    // Use Celeritas "hit processor" to call back to Geant4 SDs.
    sd.enabled = true;

    // Only call back for nonzero energy depositions: this is currently a
    // global option for all detectors, so if any SDs extract data from tracks
    // with no local energy deposition over the step, it must be set to false.
    sd.ignore_zero_deposition = true;

    // Using the pre-step point, reconstruct the G4 touchable handle.
    sd.locate_touchable = true;

    // Since at least one SD uses the pre-step time, export it from Celeritas.
    sd.pre.global_time = true;
}

//---------------------------------------------------------------------------//
/*!
 * Load geometry and sensitive detector volumes.
 */
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    CELER_LOG_LOCAL(status) << "Loading detector geometry";

    G4GDMLParser gdml_parser;
    gdml_parser.SetStripFlag(true);
    if (!GlobalSetup::Instance()->StripGDMLPointers())
    {
        // DEPRECATED: remove in 1.0?
        CELER_LOG(warning) << "Ignoring deprecated 'stripGDMLPointers false'";
    }

    std::string const& filename = GlobalSetup::Instance()->GetGeometryFile();
    if (filename.empty())
    {
        G4Exception("DetectorConstruction::Construct()",
                    "",
                    FatalException,
                    "No GDML file was specified with setGeometryFile");
    }
    constexpr bool validate_gdml_schema = false;
    gdml_parser.Read(filename, validate_gdml_schema);

    // Find sensitive detectors
    for (auto const& lv_vecaux : *gdml_parser.GetAuxMap())
    {
        for (G4GDMLAuxStructType const& aux : lv_vecaux.second)
        {
            if (aux.type == "SensDet")
            {
                detectors_.insert({aux.value, lv_vecaux.first});
            }
        }
    }

    // Claim ownership of world volume and pass it to the caller
    return gdml_parser.GetWorldVolume();
}

//---------------------------------------------------------------------------//
void DetectorConstruction::ConstructSDandField()
{
    CELER_LOG_LOCAL(status) << "Loading sensitive detectors";

    G4SDManager* sd_manager = G4SDManager::GetSDMpointer();

    auto iter = detectors_.begin();
    while (iter != detectors_.end())
    {
        // Find the end of the current range of keys
        auto stop = iter;
        do
        {
            ++stop;
        } while (stop != detectors_.end() && iter->first == stop->first);

        // Create one detector for all the volumes
        auto detector = std::make_unique<SensitiveDetector>(iter->first);

        // Attach sensitive detectors
        for (; iter != stop; ++iter)
        {
            CELER_LOG_LOCAL(debug) << "Attaching " << iter->first << " to "
                                   << iter->second->GetName();
            iter->second->SetSensitiveDetector(detector.get());
        }

        // Hand SD to the manager
        sd_manager->AddNewDetector(detector.release());
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
