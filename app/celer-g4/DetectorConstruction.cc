//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include <map>
#include <G4ChordFinder.hh>
#include <G4Exception.hh>
#include <G4FieldManager.hh>
#include <G4GDMLAuxStructType.hh>
#include <G4GDMLParser.hh>
#include <G4LogicalVolume.hh>
#include <G4MagneticField.hh>
#include <G4SDManager.hh>
#include <G4TransportationManager.hh>
#include <G4UniformMagField.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/io/Logger.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "accel/AlongStepFactory.hh"
#include "accel/RZMapMagneticField.hh"
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
                    "No GDML file was specified with /celerg4/geometryFile");
    }
    constexpr bool validate_gdml_schema = false;
    gdml_parser.Read(filename, validate_gdml_schema);

    auto& sd = celeritas::app::GlobalSetup::Instance()->GetSDSetupOptions();
    if (sd.enabled)
    {
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

        if (detectors_.empty())
        {
            CELER_LOG(warning) << "No sensitive detectors were found in the "
                                  "GDML file";
            sd.enabled = false;
        }
    }

    // Setup options for the magnetic field
    auto field_type = GlobalSetup::Instance()->GetFieldType();
    if (field_type == "rzmap")
    {
        auto map_filename = GlobalSetup::Instance()->GetFieldFile();
        if (map_filename.empty())
        {
            G4Exception("DetectorConstruction::Construct()",
                        "",
                        FatalException,
                        "No field file was specified with /celerg4/fieldFile");
        }
        CELER_LOG_LOCAL(info) << "Using RZMapField with " << map_filename;

        // Create celeritas::RZMapFieldParams from input
        RZMapFieldInput rz_map;
        std::ifstream(map_filename) >> rz_map;
        rz_map.driver_options = GlobalSetup::Instance()->GetFieldOptions();
        field_params_ = std::make_shared<RZMapFieldParams>(rz_map);
        mag_field_ = std::make_shared<RZMapMagneticField>(field_params_);

        GlobalSetup::Instance()->SetAlongStepFactory(
            RZMapFieldAlongStepFactory([=] { return rz_map; }));
    }
    else if (field_type == "uniform")
    {
        auto field = GlobalSetup::Instance()->GetMagFieldZTesla();
        if (norm(field) > 0)
        {
            CELER_LOG_LOCAL(info)
                << "Using a uniform field (0, 0, " << field[2] << ") in tesla";
            mag_field_ = std::make_shared<G4UniformMagField>(
                convert_to_geant(field, CLHEP::tesla));
        }

        // Convert field units from tesla to native celeritas units
        for (real_type& v : field)
        {
            v *= units::tesla;
        }

        UniformFieldParams input;
        input.field = field;
        input.options = GlobalSetup::Instance()->GetFieldOptions();
        GlobalSetup::Instance()->SetAlongStepFactory(
            UniformAlongStepFactory([=] { return input; }));
    }
    else
    {
        CELER_VALIDATE(false, << "invalid field type '" << field_type << "'");
    }

    // Claim ownership of world volume and pass it to the caller
    return gdml_parser.GetWorldVolume();
}

//---------------------------------------------------------------------------//
void DetectorConstruction::ConstructSDandField()
{
    if (mag_field_)
    {
        // Create the chord finder with the driver parameters
        auto const& field_options = GlobalSetup::Instance()->GetFieldOptions();
        auto chord_finder = std::make_unique<G4ChordFinder>(
            mag_field_.get(),
            convert_to_geant(field_options.minimum_step, CLHEP::cm));
        chord_finder->SetDeltaChord(
            convert_to_geant(field_options.delta_chord, CLHEP::cm));

        // Construct the magnetic field
        G4FieldManager* field_manager
            = G4TransportationManager::GetTransportationManager()
                  ->GetFieldManager();
        field_manager->SetDetectorField(mag_field_.get());
        field_manager->SetChordFinder(chord_finder.release());
        field_manager->SetMinimumEpsilonStep(field_options.epsilon_step);
        field_manager->SetDeltaIntersection(
            convert_to_geant(field_options.delta_intersection, CLHEP::cm));
    }

    if (detectors_.empty())
    {
        return;
    }

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
            CELER_LOG_LOCAL(debug)
                << "Attaching '" << iter->first << "'@" << detector.get()
                << " to '" << iter->second->GetName() << "'@"
                << static_cast<void const*>(iter->second);
            iter->second->SetSensitiveDetector(detector.get());
        }

        // Hand SD to the manager
        sd_manager->AddNewDetector(detector.release());
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
