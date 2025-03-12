//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/GlobalSetup.cc
//---------------------------------------------------------------------------//
#include "GlobalSetup.hh"

#include <utility>
#include <G4GenericMessenger.hh>
#include <G4UImanager.hh>
#include <nlohmann/json.hpp>

#include "corecel/Assert.hh"
#include "corecel/io/FileOrConsole.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "accel/SetupOptionsMessenger.hh"

#include "RunInputIO.json.hh"

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
 *
 * \deprecated Macro support for celer-g4 is deprecated: it will only take JSON
 * in the future.
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
    if (!ends_with(filename, ".mac"))
    {
        FileOrStdin instream(filename);
        CELER_LOG(status) << "Reading JSON input from '" << instream.filename()
                          << "'";
        nlohmann::json::parse(instream).get_to(input_);

        if (input_.cuda_stack_size != RunInput::unspecified)
        {
            options_->cuda_stack_size = input_.cuda_stack_size;
        }
        if (input_.cuda_heap_size != RunInput::unspecified)
        {
            options_->cuda_heap_size = input_.cuda_heap_size;
        }
        celeritas::environment().merge(input_.environ);

        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            static char const fi_hack_envname[] = "ORANGE_FORCE_INPUT";
            auto const& filename = celeritas::getenv(fi_hack_envname);
            if (!filename.empty())
            {
                CELER_LOG(warning)
                    << "Using a temporary, unsupported, and dangerous hack to "
                       "override the ORANGE geometry file: "
                    << fi_hack_envname << "='" << filename << "'";
                options_->geometry_file = filename;
            }
        }

        // Output options
        options_->output_file = input_.output_file;
        options_->physics_output_file = input_.physics_output_file;
        options_->offload_output_file = input_.offload_output_file;

        // Apply Celeritas \c SetupOptions commands
        options_->max_num_tracks = input_.num_track_slots;
        CELER_VALIDATE(input_.primary_options || !input_.event_file.empty(),
                       << "no event input file nor primary options were "
                          "specified");
        options_->max_steps = input_.max_steps;
        options_->initializer_capacity = input_.initializer_capacity;
        options_->secondary_stack_factor = input_.secondary_stack_factor;
        options_->auto_flush = input_.auto_flush;

        options_->max_field_substeps = input_.field_options.max_substeps;

        options_->sd.enabled = input_.sd_type != SensitiveDetectorType::none;
        options_->slot_diagnostic_prefix = input_.slot_diagnostic_prefix;

        options_->action_times = input_.action_times;
        options_->default_stream = input_.default_stream;
        options_->track_order = input_.track_order;
    }
    else
    {
        input_.macro_file = filename;
    }

    // Execute macro for Geant4 commands (e.g. to set verbosity)
    if (!input_.macro_file.empty())
    {
        CELER_LOG(status) << "Executing macro commands from '" << filename
                          << "'";
        G4UImanager* ui = G4UImanager::GetUIpointer();
        CELER_ASSERT(ui);
        ui->ApplyCommand("/control/execute " + input_.macro_file);
    }

    // Set the filename for JSON output
    if (input_.output_file.empty())
    {
        input_.output_file = "celer-g4.out.json";
        options_->output_file = input_.output_file;
    }

    if (input_.sd_type == SensitiveDetectorType::event_hit)
    {
        root_sd_io_ = RootFileManager::use_root();
        if (!root_sd_io_)
        {
            CELER_LOG(warning) << "Collecting SD hit data that will not be "
                                  "written because ROOT is disabled";
        }
    }
    else
    {
        root_sd_io_ = false;
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
