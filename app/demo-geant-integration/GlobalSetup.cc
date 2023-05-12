//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/GlobalSetup.cc
//---------------------------------------------------------------------------//
#include "GlobalSetup.hh"

#include <utility>
#include <G4GenericMessenger.hh>

#include "corecel/Assert.hh"
#include "corecel/sys/Device.hh"
#include "accel/AlongStepFactory.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Return non-owning pointer to a singleton.
 */
GlobalSetup* GlobalSetup::Instance()
{
    static GlobalSetup setup;
    return &setup;
}

//---------------------------------------------------------------------------//
/*!
 * Set configurable properties from the UI.
 */
GlobalSetup::GlobalSetup()
{
    options_ = std::make_shared<SetupOptions>();
    field_ = G4ThreeVector(0, 0, 0);
    messenger_ = std::make_unique<G4GenericMessenger>(
        this, "/setup/", "Demo geant integration setup");

    {
        auto& cmd
            = messenger_->DeclareProperty("geometryFile", geometry_file_);
        cmd.SetGuidance("Set the filename of the GDML detector geometry");
    }
    {
        auto& cmd = messenger_->DeclareProperty("eventFile", event_file_);
        cmd.SetGuidance("Set the filename of the event input read by HepMC3");
    }
    {
        auto& cmd = messenger_->DeclareProperty("rootBufferSize",
                                                root_buffer_size_);
        cmd.SetGuidance("Set the buffer size (bytes) of output root file");
        cmd.SetDefaultValue(std::to_string(root_buffer_size_));
    }
    {
        auto& cmd = messenger_->DeclareProperty("writeSDHits",
                                                write_sd_hits_);
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
        auto& cmd = messenger_->DeclareProperty("outputFile",
                                                options_->output_file);
        cmd.SetGuidance("Set the JSON output file name");
    }
    {
        auto& cmd = messenger_->DeclareProperty("maxNumTracks",
                                                options_->max_num_tracks);
        cmd.SetGuidance("Set the maximum number of track slots");
        options_->max_num_tracks
            = celeritas::Device::num_devices() > 0 ? 524288 : 64;
        cmd.SetDefaultValue(std::to_string(options_->max_num_tracks));
    }
    {
        auto& cmd = messenger_->DeclareProperty("maxNumEvents",
                                                options_->max_num_events);
        cmd.SetGuidance("Set the maximum number of events in the run");
        options_->max_num_events = 1024;
        cmd.SetDefaultValue(std::to_string(options_->max_num_events));
    }
    {
        auto& cmd = messenger_->DeclareProperty(
            "secondaryStackFactor", options_->secondary_stack_factor);
        cmd.SetGuidance("Set the number of secondary slots per track slot");
        options_->secondary_stack_factor = 3;
        cmd.SetDefaultValue(std::to_string(options_->secondary_stack_factor));
    }
    {
        auto& cmd = messenger_->DeclareProperty(
            "initializerCapacity", options_->initializer_capacity);
        cmd.SetGuidance("Set the maximum number of queued tracks");
        options_->initializer_capacity = 1048576;
        cmd.SetDefaultValue(std::to_string(options_->initializer_capacity));
    }
    {
        auto& cmd = messenger_->DeclareProperty("cudaStackSize",
                                                options_->cuda_stack_size);
        cmd.SetGuidance("Set the per-thread dynamic CUDA stack size (bytes)");
        options_->cuda_stack_size = 0;
    }
    {
        auto& cmd = messenger_->DeclareProperty("cudaHeapSize",
                                                options_->cuda_heap_size);
        cmd.SetGuidance("Set the shared dynamic CUDA heap size (bytes)");
        options_->cuda_heap_size = 0;
    }
    {
        messenger_->DeclareMethod("magFieldZ",
                                  &GlobalSetup::SetMagFieldZTesla,
                                  "Set Z-axis magnetic field strength (T)");
    }
    {
        // TODO: expose other options here
    }

    // At setup time, get the field strength (native G4units)
    options_->make_along_step
        = celeritas::UniformAlongStepFactory([this] { return field_; });
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
}  // namespace demo_geant
