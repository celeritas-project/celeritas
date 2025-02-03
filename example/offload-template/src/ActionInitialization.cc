//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file example/offload-template/src/ActionInitialization.cc
//---------------------------------------------------------------------------//
#include "ActionInitialization.hh"

#include <accel/SetupOptions.hh>
#include <accel/TrackingManagerIntegration.hh>

#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"

//---------------------------------------------------------------------------//
/*!
 * Construct empty.
 */
ActionInitialization::ActionInitialization() : G4VUserActionInitialization()
{
    // Initialize Celeritas
    celeritas::SetupOptions& so
        = celeritas::TrackingManagerIntegration::Instance().Options();

    so.max_num_tracks = 1024 * 16;
    so.initializer_capacity = 1024 * 128 * 4;
    so.secondary_stack_factor = 2.0;
    so.ignore_processes = {"CoulombScat", "Rayl"};  // Ignored processes

    // Save diagnostic information
    so.output_file = "celeritas-offload-diagnostic.json";
}

//---------------------------------------------------------------------------//
/*!
 * Set up Celeritas offload on master thread and initialize it via the
 * \c G4UserRunAction .
 */
void ActionInitialization::BuildForMaster() const
{
    // Set up Celeritas integration
    celeritas::TrackingManagerIntegration::Instance().BuildForMaster();

    // RunAction is responsible for initializing Celeritas
    this->SetUserAction(new RunAction());
}

//---------------------------------------------------------------------------//
/*!
 * Set up all worker thread user actions and Celeritas offload interface.
 */
void ActionInitialization::Build() const
{
    // Set up Celeritas integration
    celeritas::TrackingManagerIntegration::Instance().Build();

    // Initialize Geant4 user actions
    this->SetUserAction(new RunAction());
    this->SetUserAction(new PrimaryGeneratorAction());
}
