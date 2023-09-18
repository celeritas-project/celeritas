//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GlobalSetup.cc
//---------------------------------------------------------------------------//
#include "GlobalSetup.hh"

#include <fstream>
#include <utility>
#include <G4GenericMessenger.hh>
#include <G4UImanager.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "accel/AlongStepFactory.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/SetupOptionsMessenger.hh"

#include "HepMC3PrimaryGeneratorAction.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "RunInputIO.json.hh"
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Return non-owning pointer to a singleton.
 *
 * Creating the instance also creates a "messenger" that allows control over
 * the Celeritas user inputs.
 */
GlobalSetup* GlobalSetup::Instance()
{
    static GlobalSetup setup;
    static SetupOptionsMessenger mess{setup.options_.get()};
    return &setup;
}

//---------------------------------------------------------------------------//
/*!
 * Set configurable properties from the UI.
 */
GlobalSetup::GlobalSetup()
{
    options_ = std::make_shared<SetupOptions>();

    messenger_ = std::make_unique<G4GenericMessenger>(
        this, "/celerg4/", "Demo geant integration setup");

    {
        auto& cmd = messenger_->DeclareProperty("geometryFile",
                                                input_.geometry_file);
        cmd.SetGuidance("Filename of the GDML detector geometry");
    }
    {
        auto& cmd = messenger_->DeclareProperty("eventFile", input_.event_file);
        cmd.SetGuidance("Filename of the event input read by HepMC3");
    }
    {
        auto& cmd = messenger_->DeclareProperty("rootBufferSize",
                                                input_.root_buffer_size);
        cmd.SetGuidance("Buffer size of output root file [bytes]");
        cmd.SetDefaultValue(std::to_string(input_.root_buffer_size));
    }
    {
        auto& cmd
            = messenger_->DeclareProperty("writeSDHits", input_.write_sd_hits);
        cmd.SetGuidance("Write a ROOT output file with hits from the SDs");
        cmd.SetDefaultValue("false");
    }
    {
        auto& cmd = messenger_->DeclareProperty("stripGDMLPointers",
                                                input_.strip_gdml_pointers);
        cmd.SetGuidance(
            "Remove pointer suffix from input logical volume names");
        cmd.SetDefaultValue("true");
    }
    {
        auto& cmd = messenger_->DeclareProperty("stepDiagnostic",
                                                input_.step_diagnostic);
        cmd.SetGuidance("Collect the distribution of steps per Geant4 track");
        cmd.SetDefaultValue("false");
    }
    {
        auto& cmd = messenger_->DeclareProperty("stepDiagnosticBins",
                                                input_.step_diagnostic_bins);
        cmd.SetGuidance("Number of bins for the Geant4 step diagnostic");
        cmd.SetDefaultValue(std::to_string(input_.step_diagnostic_bins));
    }
    // Setup options for the magnetic field
    {
        auto& cmd = messenger_->DeclareProperty("fieldType", input_.field_type);
        cmd.SetGuidance("Select the field type [rzmap|uniform]");
        cmd.SetDefaultValue(input_.field_type);
    }
    {
        auto& cmd = messenger_->DeclareProperty("fieldFile", input_.field_file);
        cmd.SetGuidance("Filename of the rz-map loaded by RZMapFieldInput");
    }
    {
        messenger_->DeclareMethod("magFieldZ",
                                  &GlobalSetup::SetMagFieldZTesla,
                                  "Set Z-axis magnetic field strength (T)");
    }
}

//---------------------------------------------------------------------------//
/*!
 * Set the list of ignored EM process names.
 */
void GlobalSetup::SetIgnoreProcesses(SetupOptions::VecString ignored)
{
    options_->ignore_processes = std::move(ignored);
}

//---------------------------------------------------------------------------//
/*!
 * Read input from macro or JSON.
 */
void GlobalSetup::ReadInput(std::string const& filename)
{
    if (ends_with(filename, ".mac"))
    {
        CELER_LOG(status) << "Executing macro commands from '" << filename
                          << "'";
        G4UImanager* ui = G4UImanager::GetUIpointer();
        CELER_ASSERT(ui);
        ui->ApplyCommand(std::string("/control/execute ") + filename);
    }
    else
    {
#if CELERITAS_USE_JSON
        using std::to_string;

        CELER_LOG(status) << "Reading JSON input from '" << filename << "'";
        std::ifstream infile(filename);
        CELER_VALIDATE(infile, << "failed to open '" << filename << "'");
        nlohmann::json::parse(infile).get_to(input_);
        CELER_ASSERT(input_);

        // Input options
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            // To allow ORANGE to work for testing purposes, pass the GDML
            // input filename to Celeritas
            options_->geometry_file = input_.geometry_file;
        }

        // Output options
        options_->output_file = input_.output_file;
        options_->physics_output_file = input_.physics_output_file;
        options_->offload_output_file = input_.offload_output_file;

        // Apply Celeritas \c SetupOptions commands
        options_->max_num_tracks = input_.num_track_slots;
        options_->max_num_events = input_.max_events;
        options_->max_steps = input_.max_steps;
        options_->initializer_capacity = input_.initializer_capacity;
        options_->secondary_stack_factor = input_.secondary_stack_factor;
        options_->sd.enabled = input_.enable_sd;
        options_->cuda_stack_size = input_.cuda_stack_size;
        options_->cuda_heap_size = input_.cuda_heap_size;
        options_->sync = input_.sync;
        options_->default_stream = input_.default_stream;

        // Execute macro for Geant4 commands (e.g. to set verbosity)
        if (!input_.macro_file.empty())
        {
            G4UImanager* ui = G4UImanager::GetUIpointer();
            CELER_ASSERT(ui);
            ui->ApplyCommand("/control/execute " + input_.macro_file);
        }
#else
        CELER_NOT_CONFIGURED("nlohmann_json");
#endif
    }

    // Set the filename for JSON output
    if (input_.output_file.empty())
    {
        input_.output_file = "celer-g4.out.json";
        options_->output_file = input_.output_file;
    }

    // Get the number of events
    if (!input_.event_file.empty())
    {
        // Load the input file
        CELER_TRY_HANDLE(
            num_events_ = HepMC3PrimaryGeneratorAction::NumEvents(),
            ExceptionConverter{"celer-g4000"});
    }
    else
    {
        num_events_ = input_.primary_options.num_events;
    }

    // Start the timer for setup time
    get_setup_time_ = {};
}

//---------------------------------------------------------------------------//
//! Default destructor
GlobalSetup::~GlobalSetup() = default;

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
