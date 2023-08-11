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
#include "corecel/sys/Device.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "accel/AlongStepFactory.hh"
#include "accel/SetupOptionsMessenger.hh"

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
 * Read input from JSON.
 */
void GlobalSetup::ReadInput(std::string const& filename)
{
#if CELERITAS_USE_JSON
    using std::to_string;

    std::ifstream infile(filename);
    CELER_VALIDATE(infile, << "failed to open '" << filename << "'");
    nlohmann::json::parse(infile).get_to(input_);
    CELER_ASSERT(input_);

    // Apply Celeritas \c SetupOptions commands
    G4UImanager* ui = G4UImanager::GetUIpointer();
    CELER_ASSERT(ui);
    ui->ApplyCommand("/celer/maxNumTracks "
                     + to_string(input_.num_track_slots));
    ui->ApplyCommand("/celer/maxNumEvents " + to_string(input_.max_events));
    ui->ApplyCommand("/celer/maxNumSteps " + to_string(input_.max_steps));
    ui->ApplyCommand("/celer/maxInitializers "
                     + to_string(input_.initializer_capacity));
    ui->ApplyCommand("/celer/secondaryStackFactor "
                     + to_string(input_.secondary_stack_factor));
    ui->ApplyCommand("/celer/cuda/stackSize "
                     + to_string(input_.cuda_stack_size));
    ui->ApplyCommand("/celer/cuda/heapSize "
                     + to_string(input_.cuda_heap_size));
    ui->ApplyCommand("/celer/cuda/sync" + to_string(input_.sync));
    ui->ApplyCommand("/celer/cuda/defaultStream"
                     + to_string(input_.default_stream));
    ui->ApplyCommand("/celer/outputFile " + input_.output_file);
    ui->ApplyCommand("/celer/offloadFile " + input_.offload_file);

    // Execute macro for Geant4 commands (e.g. to set verbosity)
    if (!input_.macro_file.empty())
    {
        ui->ApplyCommand("/control/execute " + input_.macro_file);
    }
#else
    CELER_NOT_CONFIGURED("nlohmann_json");
#endif
}

//---------------------------------------------------------------------------//
//! Default destructor
GlobalSetup::~GlobalSetup() = default;

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
