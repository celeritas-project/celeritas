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

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "accel/AlongStepFactory.hh"
#include "accel/SetupOptionsMessenger.hh"

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
        auto& cmd = messenger_->DeclareProperty("geometryFile", geometry_file_);
        cmd.SetGuidance("Filename of the GDML detector geometry");
    }
    {
        auto& cmd = messenger_->DeclareProperty("eventFile", event_file_);
        cmd.SetGuidance("Filename of the event input read by HepMC3");
    }
    {
        auto& cmd
            = messenger_->DeclareProperty("rootBufferSize", root_buffer_size_);
        cmd.SetGuidance("Buffer size of output root file [bytes]");
        cmd.SetDefaultValue(std::to_string(root_buffer_size_));
    }
    {
        auto& cmd = messenger_->DeclareProperty("writeSDHits", write_sd_hits_);
        cmd.SetGuidance("Write a ROOT output file with hits from the SDs");
        cmd.SetDefaultValue("false");
    }
    {
        auto& cmd = messenger_->DeclareProperty("stripGDMLPointers",
                                                strip_gdml_pointers_);
        cmd.SetGuidance(
            "Remove pointer suffix from input logical volume names");
        cmd.SetDefaultValue("true");
    }
    {
        auto& cmd = messenger_->DeclareProperty("physicsList", physics_list_);
        cmd.SetGuidance("Select the physics list");
        cmd.SetDefaultValue(physics_list_);
    }
    {
        auto& cmd
            = messenger_->DeclareProperty("stepDiagnostic", step_diagnostic_);
        cmd.SetGuidance("Collect the distribution of steps per Geant4 track");
        cmd.SetDefaultValue("false");
    }
    {
        auto& cmd = messenger_->DeclareProperty("stepDiagnosticBins",
                                                step_diagnostic_bins_);
        cmd.SetGuidance("Number of bins for the Geant4 step diagnostic");
        cmd.SetDefaultValue(std::to_string(step_diagnostic_bins_));
    }

    // Setup options for the magnetic field
    {
        auto& cmd = messenger_->DeclareProperty("fieldType", field_type_);
        cmd.SetGuidance("Select the field type [rzmap|uniform]");
        cmd.SetDefaultValue(field_type_);
    }
    {
        auto& cmd = messenger_->DeclareProperty("fieldFile", field_file_);
        cmd.SetGuidance("Filename of the rz-map loaded by RZMapFieldInput");
    }
    {
        messenger_->DeclareMethod("magFieldZ",
                                  &GlobalSetup::SetMagFieldZTesla,
                                  "Set Z-axis magnetic field strength (T)");
    }
    {
        // TODO: expose other options here
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
//! Default destructor
GlobalSetup::~GlobalSetup() = default;

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
