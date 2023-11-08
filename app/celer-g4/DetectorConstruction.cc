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

#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "accel/AlongStepFactory.hh"
#include "accel/RZMapMagneticField.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"

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
DetectorConstruction::DetectorConstruction(SPParams params)
    : params_{std::move(params)}
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
    auto geo = this->construct_geo();
    CELER_ASSERT(geo.world);
    detectors_ = std::move(geo.detectors);

    auto field = this->construct_field();
    CELER_ASSERT(field.along_step);
    GlobalSetup::Instance()->SetAlongStepFactory(std::move(field.along_step));
    mag_field_ = std::move(field.g4field);
    return geo.world.release();
}

//---------------------------------------------------------------------------//
/*!
 * Construct shared geometry information.
 */
auto DetectorConstruction::construct_geo() const -> GeoData
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

    MapDetectors detectors;
    if (sd.enabled)
    {
        // Find sensitive detectors
        for (auto const& lv_vecaux : *gdml_parser.GetAuxMap())
        {
            for (G4GDMLAuxStructType const& aux : lv_vecaux.second)
            {
                if (aux.type == "SensDet")
                {
                    detectors.insert({aux.value, lv_vecaux.first});
                }
            }
        }

        if (detectors.empty())
        {
            CELER_LOG(warning) << "No sensitive detectors were found in the "
                                  "GDML file";
            sd.enabled = false;
        }
    }

    // Claim ownership of world volume and pass it to the caller
    return {std::move(detectors),
            UPPhysicalVolume{gdml_parser.GetWorldVolume()}};
}

//---------------------------------------------------------------------------//
/*!
 * Construct shared magnetic field information.
 *
 * This creates the shared Celeritas object and saves some field information
 * for later.
 */
auto DetectorConstruction::construct_field() const -> FieldData
{
    // Set up Celeritas magnetic field and save for Geant4
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
        auto rz_map = [&map_filename] {
            RZMapFieldInput rz_map;
            std::ifstream inp(map_filename);
            CELER_VALIDATE(inp,
                           << "failed to open field map file at '"
                           << map_filename << "'");
            inp >> rz_map;
            return rz_map;
        }();

        // Replace driver options with user options
        rz_map.driver_options = GlobalSetup::Instance()->GetFieldOptions();
        auto field_params = std::make_shared<RZMapFieldParams>(rz_map);

        // Return celeritas and geant4 fields
        return {RZMapFieldAlongStepFactory([rz_map] { return rz_map; }),
                std::make_shared<RZMapMagneticField>(std::move(field_params))};
    }
    else if (field_type == "uniform")
    {
        SPMagneticField g4field;
        auto field_val = GlobalSetup::Instance()->GetMagFieldTesla();
        if (norm(field_val) > 0)
        {
            CELER_LOG_LOCAL(info)
                << "Using a uniform field " << field_val << " [tesla]";
            g4field = std::make_shared<G4UniformMagField>(
                convert_to_geant(field_val, CLHEP::tesla));
        }

        // Convert field units from tesla to native celeritas units
        for (real_type& v : field_val)
        {
            v = native_value_from(units::FieldTesla{v});
        }

        UniformFieldParams input;
        input.field = field_val;
        input.options = GlobalSetup::Instance()->GetFieldOptions();

        return {UniformAlongStepFactory([input] { return input; }),
                std::move(g4field)};
    }

    CELER_VALIDATE(false, << "invalid field type '" << field_type << "'");
}

//---------------------------------------------------------------------------//
/*!
 * Construct thread-local sensitive detectors and field.
 */
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
