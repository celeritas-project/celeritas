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
    if (!input_file_.empty())
    {
#if CELERITAS_USE_JSON
        // Open the auxiliary JSON input file
        std::ifstream infile(input_file_);
        CELER_VALIDATE(infile, << "failed to open '" << input_file_ << "'");
        nlohmann::json::parse(infile).get_to(input_);
#else
        CELER_NOT_CONFIGURED("nlohmann_json");
#endif
    }

    options_ = std::make_shared<SetupOptions>();

    messenger_ = std::make_unique<G4GenericMessenger>(
        this, "/celerg4/", "Demo geant integration setup");

    {
        auto& cmd = messenger_->DeclareProperty("inputFile", input_file_);
        cmd.SetGuidance("Filename of the auxiliary JSON input");
    }
    {
        auto& cmd = messenger_->DeclareProperty("geometryFile", geometry_file_);
        cmd.SetGuidance("Filename of the GDML detector geometry");
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
