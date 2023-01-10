//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
    messenger_ = std::make_unique<G4GenericMessenger>(
        this, "/setup/", "Demo geant integration setup");

    {
        auto& cmd
            = messenger_->DeclareProperty("setGeometryFile", geometry_file_);
        cmd.SetGuidance("Set the filename of the GDML detector geometry");
    }
    {
        auto& cmd = messenger_->DeclareProperty("setEventFile", event_file_);
        cmd.SetGuidance("Set the filename of the event input read by HepMC3");
    }
    {
        auto& cmd = messenger_->DeclareProperty("setOutputFile",
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
        // TODO: expose other options here
    }
}

//---------------------------------------------------------------------------//
/*!
 * Set the along-step factory.
 *
 * The "asf" can be an instance of a class inheriting from \c
 * celeritas::AlongStepFactoryInterface , a functor, or just a function.
 */
void GlobalSetup::SetAlongStep(SetupOptions::AlongStepFactory asf)
{
    CELER_EXPECT(asf);

    options_->make_along_step = std::move(asf);
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
