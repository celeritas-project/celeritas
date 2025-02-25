//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/DetectorConstruction.cc
//---------------------------------------------------------------------------//
#include "DetectorConstruction.hh"

#include <map>
#include <G4ChordFinder.hh>
#include <G4Exception.hh>
#include <G4FieldManager.hh>
#include <G4LogicalVolume.hh>
#include <G4MagneticField.hh>
#include <G4SDManager.hh>
#include <G4TransportationManager.hh>
#include <G4UniformMagField.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantGdmlLoader.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "accel/AlongStepFactory.hh"
#include "accel/GeantSimpleCalo.hh"
#include "accel/RZMapMagneticField.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"

#include "GeantDiagnostics.hh"
#include "GlobalSetup.hh"
#include "RootIO.hh"
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
    auto& sd = GlobalSetup::Instance()->GetSDSetupOptions();

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
 *
 * This should only be called once from the master thread, toward the very
 * beginning of the program.
 */
G4VPhysicalVolume* DetectorConstruction::Construct()
{
    CELER_EXPECT(GlobalSetup::Instance()->GetSDSetupOptions().enabled
                 == (GlobalSetup::Instance()->input().sd_type
                     != SensitiveDetectorType::none));

    std::string const& filename = GlobalSetup::Instance()->GetGeometryFile();
    CELER_VALIDATE(!filename.empty(),
                   << "no GDML input file was specified (geometry_file)");

    auto& sd = GlobalSetup::Instance()->GetSDSetupOptions();
    auto loaded = [&] {
        GeantGdmlLoader::Options opts;
        opts.detectors = sd.enabled;
        GeantGdmlLoader load(opts);
        return load(filename);
    }();

    if (sd.enabled && loaded.detectors.empty())
    {
        CELER_LOG(warning)
            << R"(No sensitive detectors were found in the GDML file)";
        sd.enabled = false;
    }

    CELER_ASSERT(loaded.world);
    detectors_ = std::move(loaded.detectors);

    if (!detectors_.empty()
        && GlobalSetup::Instance()->input().sd_type
               == SensitiveDetectorType::simple_calo)
    {
        this->foreach_detector([this](auto start, auto stop) {
            std::string name = start->first;
            std::vector<G4LogicalVolume*> volumes;
            for (auto iter = start; iter != stop; ++iter)
            {
                volumes.push_back(iter->second);
            }
            CELER_LOG(debug) << "Creating GeantSimpleCalo '" << name
                             << "' with " << volumes.size() << " volumes";

            // Create calo manager
            simple_calos_.push_back(std::make_shared<GeantSimpleCalo>(
                std::move(name), params_, std::move(volumes)));
        });
    }

    // Add outputs to the Geant diagnostics
    GeantDiagnostics::register_output(
        {simple_calos_.begin(), simple_calos_.end()});

    auto field = this->construct_field();
    CELER_ASSERT(field.along_step);
    GlobalSetup::Instance()->SetAlongStepFactory(std::move(field.along_step));
    mag_field_ = std::move(field.g4field);
    return loaded.world;
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
                << "Using a uniform field " << field_val << " [T]";
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
    CELER_ASSERT_UNREACHABLE();
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
            convert_to_geant(field_options.minimum_step, clhep_length));
        chord_finder->SetDeltaChord(
            convert_to_geant(field_options.delta_chord, clhep_length));

        // Construct the magnetic field
        G4FieldManager* field_manager
            = G4TransportationManager::GetTransportationManager()
                  ->GetFieldManager();
        field_manager->SetDetectorField(mag_field_.get());
        field_manager->SetChordFinder(chord_finder.release());
        field_manager->SetMinimumEpsilonStep(field_options.epsilon_step);
        field_manager->SetDeltaIntersection(
            convert_to_geant(field_options.delta_intersection, clhep_length));
    }

    auto sd_type = GlobalSetup::Instance()->input().sd_type;
    auto* sd_manager = G4SDManager::GetSDMpointer();

    if (sd_type == SensitiveDetectorType::none)
    {
        CELER_LOG_LOCAL(debug) << "No sensitive detectors requested";
    }
    else if (sd_type == SensitiveDetectorType::simple_calo)
    {
        for (auto& calo : simple_calos_)
        {
            CELER_LOG_LOCAL(status)
                << "Attaching simple calorimeter '" << calo->label() << '\'';
            // Create and attach SD
            auto detector = calo->MakeSensitiveDetector();
            CELER_ASSERT(detector);
            sd_manager->AddNewDetector(detector.release());
        }
    }
    else if (sd_type == SensitiveDetectorType::event_hit)
    {
        CELER_LOG_LOCAL(status) << "Creating SDs";
        this->foreach_detector([sd_manager](auto start, auto stop) {
            // Create one detector for all the volumes
            auto detector = std::make_unique<SensitiveDetector>(start->first);

            // Attach sensitive detectors
            for (auto iter = start; iter != stop; ++iter)
            {
                CELER_LOG_LOCAL(debug)
                    << "Attaching '" << iter->first << "'@" << detector.get()
                    << " to '" << iter->second->GetName() << "'@"
                    << static_cast<void const*>(iter->second);
                iter->second->SetSensitiveDetector(detector.get());
            }

            // Hand SD to the manager
            sd_manager->AddNewDetector(detector.release());

            // Add to ROOT output
            if (GlobalSetup::Instance()->root_sd_io())
            {
                RootIO::Instance()->AddSensitiveDetector(start->first);
            }
        });
    }
}

//---------------------------------------------------------------------------//
/*!
 * Apply a function to the range of volumes for each detector.
 */
template<class F>
void DetectorConstruction::foreach_detector(F&& apply_to_range) const
{
    auto start = detectors_.begin();
    while (start != detectors_.end())
    {
        // Find the end of the current range of keys
        auto stop = start;
        do
        {
            ++stop;
        } while (stop != detectors_.end() && start->first == stop->first);

        apply_to_range(start, stop);
        start = stop;
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
