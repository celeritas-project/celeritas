//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/GlobalSetup.cc
//---------------------------------------------------------------------------//
#include "GlobalSetup.hh"

#include <G4GenericMessenger.hh>

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
    options_   = std::make_shared<celeritas::SetupOptions>();
    messenger_ = std::make_unique<G4GenericMessenger>(
        this, "/setup/", "Demo geant integration setup");

    {
        auto& cmd = messenger_->DeclareProperty("setGeometryFile",
                                                options_->geometry_file);
        cmd.SetGuidance("Set the geometry file name");
        cmd.SetDefaultValue("");
    }
    {
        auto& cmd = messenger_->DeclareProperty("maxNumTracks",
                                                options_->max_num_tracks);
        cmd.SetGuidance("Set the maximum number of track slots");
        cmd.SetDefaultValue(celeritas::Device::num_devices() > 0 ? "524288"
                                                                 : "64");
    }
    {
        auto& cmd = messenger_->DeclareProperty("maxNumEvents",
                                                options_->max_num_events);
        cmd.SetGuidance("Set the maximum number of events in the run");
        cmd.SetDefaultValue("1024");
    }
    {
        auto& cmd = messenger_->DeclareProperty(
            "secondaryStackFactor", options_->secondary_stack_factor);
        cmd.SetGuidance("Set the number of secondary slots per track slot");
        cmd.SetDefaultValue("3");
    }
    {
        auto& cmd = messenger_->DeclareProperty(
            "initializerCapacity", options_->initializer_capacity);
        cmd.SetGuidance("Set the maximum number of queued tracks");
        cmd.SetDefaultValue("1048576");
    }
    {
        auto& cmd
            = messenger_->DeclareProperty("cudaStackSize", cuda_stack_size_);
        cmd.SetGuidance("Set the per-thread dynamic CUDA stack size");
        cmd.SetDefaultValue("0");
        cmd.SetUnit("B");
    }
    {
        auto& cmd
            = messenger_->DeclareProperty("cudaHeapSize", cuda_heap_size_);
        cmd.SetGuidance("Set the shared dynamic CUDA heap size");
        cmd.SetDefaultValue("0");
        cmd.SetUnit("B");
    }
    {
        // TODO: expose other options here
    }
}

//---------------------------------------------------------------------------//
//! Default destructor
GlobalSetup::~GlobalSetup() = default;

//---------------------------------------------------------------------------//
} // namespace demo_geant
