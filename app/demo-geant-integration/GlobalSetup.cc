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
#include "accel/Logger.hh"

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
    }
    {
        auto& cmd = messenger_->DeclareProperty("maxNumTracks",
                                                options_->max_num_tracks);
        cmd.SetGuidance("Set the maximum number of track slots");
    }
    {
        auto& cmd = messenger_->DeclareProperty("maxNumEvents",
                                                options_->max_num_events);
        cmd.SetGuidance("Set the maximum number of events in the run");
    }
    {
        auto& cmd = messenger_->DeclareProperty(
            "secondaryStackFactor", options_->secondary_stack_factor);
        cmd.SetGuidance("Set the number of secondary slots per track slot");
    }
    {
        auto& cmd = messenger_->DeclareProperty(
            "initializerCapacity", options_->initializer_capacity);
        cmd.SetGuidance("Set the maximum number of queued tracks");
    }
    {
        auto& cmd
            = messenger_->DeclareProperty("cudaStackSize", cuda_stack_size_);
        cmd.SetGuidance("Set the per-thread dynamic CUDA stack size (bytes)");
    }
    {
        auto& cmd
            = messenger_->DeclareProperty("cudaHeapSize", cuda_heap_size_);
        cmd.SetGuidance("Set the shared dynamic CUDA heap size (bytes)");
    }
    {
        // TODO: expose other options here
        CELER_LOG_LOCAL(debug)
            << "GlobalSetup: defaults:"
            << "\n\t geometry_file =" << this->options_->geometry_file
            << "\n\t max_num_events=" << this->options_->max_num_events
            << "\n\t max_num_tracks=" << this->options_->max_num_tracks
            << "\n\t max_steps     =" << this->options_->max_steps
            << "\n\t init_capacity =" << this->options_->initializer_capacity
            << "\n\t secStackFactor=" << this->options_->secondary_stack_factor
            << "\n\t cudaStackSize =" << this->cuda_stack_size_
            << "\n\t cudaHeapSize  =" << this->cuda_heap_size_;
    }
}

//---------------------------------------------------------------------------//
//! Default destructor
GlobalSetup::~GlobalSetup()
{
    CELER_LOG_LOCAL(debug)
        << "~GlobalSetup(): EOJ values:"
        << "\n\t geometry_file =" << this->options_->geometry_file
        << "\n\t max_num_events=" << this->options_->max_num_events
        << "\n\t max_num_tracks=" << this->options_->max_num_tracks
        << "\n\t max_steps     =" << this->options_->max_steps
        << "\n\t init_capacity =" << this->options_->initializer_capacity
        << "\n\t secStackFactor=" << this->options_->secondary_stack_factor
        << "\n\t cudaStackSize =" << this->cuda_stack_size_
        << "\n\t cudaHeapSize  =" << this->cuda_heap_size_;
}

//---------------------------------------------------------------------------//
} // namespace demo_geant
